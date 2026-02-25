# CABER: Coalgebraic LLM Behavioral Auditing — Implementation

Learns finite-state automata from black-box LLM outputs and model-checks temporal safety properties against them.

## 30-Second Quickstart

```bash
# No API key needed — run mock LLM validation
python3 phase0_experiments.py

# Build Rust core + run 38 property-based tests
cargo build --release && cargo test -p caber-integration --test property_tests

# Real LLM experiments (requires OPENAI_API_KEY)
source ~/.bashrc && python3 pathb_experiments.py
```

### Most Impressive Demo

Deploy CABER against `gpt-4.1-nano` under 3 system-prompt configurations (270 API calls):

```
$ source ~/.bashrc && python3 pathb_experiments.py

  safety_strict:       2 behavioral states, 46.7% refusal, 3/3 properties pass
  creative_permissive: 2 behavioral states, 33.3% refusal, 2/3 pass
                       → refusal persistence FAILS (model softens under pressure)
  instruction_rigid:   3 behavioral states, 60.0% refusal, 3/3 pass
                       → discovers emergent "terse" behavioral atom
  Divergent prompts:   7/15 across configs (Bayesian P(>25%) = 97.3%)

  Statistical baselines (chi-squared, MMD, KL):
    Detect aggregate divergence comparably.
    CABER adds: reusable automaton, graded satisfaction [0,1],
    temporal reasoning, Kantorovich drift metrics.
```

## Prerequisites

- **Rust** 1.75+ (2021 edition)
- **Python** 3.10+
- **openai** Python package (for real LLM experiments only)

## Project Structure

| Directory | Language | Description |
|-----------|----------|-------------|
| `caber-core/` | Rust | Core engine: coalgebra types, PCL* learner, CoalCEGAR, QCTL_F, model checker, bisimulation, audit reports |
| `caber-cli/` | Rust | Command-line interface |
| `caber-integration/` | Rust | 38 property-based tests (proptest) |
| `caber-examples/` | Rust | Runnable demos |
| `caber-python/` | Python | Mock LLMs, baselines, evaluation harness |

## Experiments

| Script | Output | API Key? | Description |
|--------|--------|----------|-------------|
| `pathb_experiments.py` | `pathb_results.json` | Yes | Statistical baselines, classifier metrics, CV, calibration, abstraction gap |
| `pathb_improvements.py` | `pathb_improvements_results.json` | No | Calibration fix (Platt/isotonic), structural advantage demo |
| `expanded_experiments.py` | `expanded_results.json` | Yes | Multi-config LLM + CoalCEGAR convergence + random baseline |
| `phase0_experiments.py` | `phase0_results.json` | No | Mock LLM validation (4 model×property configs, ≥92% accuracy) |
| `phase_c_experiments.py` | `phase_c_results.json` | No | Bayesian analysis, ablation studies, PDFA tuning |
| `classifier_robustness_analysis.py` | `classifier_robustness_results.json` | No | Monte Carlo robustness (2K trials/rate, ≥99% at ρ=0.20) |

## Key Results

| Metric | Value |
|--------|-------|
| PCL* accuracy (mock LLMs, 3–100 states) | ≥92%, scaling O(n^1.42) |
| CABER vs PDFA baseline | 5–10× more compact, +21.6pp accuracy |
| Per-atom classifier F1 (real LLM) | 0.93–1.00 (keyword) |
| Leave-one-prompt-out CV | 0.55–0.69 (limited generalization) |
| Verdict accuracy under 20% classifier error | ≥99.25% |
| CoalCEGAR convergence | 2 iterations (safety fragment proved) |
| Abstraction gap | 0–27% inconsistency (quantified per-config) |
| Calibration (after isotonic regression) | 0.13–0.21 error (from 0.28–0.73 raw) |
| Structural advantage | Detects temporal patterns invisible to χ²/MMD |

## Known Limitations

- Real LLM experiments use **gpt-4.1-nano only** (small model)
- **PAC bounds are vacuous** at 90-sample operating size (~143K required for ε=0.05); Bayesian posteriors provide non-vacuous uncertainty
- Alphabet abstraction is **non-functorial** (0–27% inconsistency)
- CoalCEGAR convergence proved for **safety fragment only**; liveness remains open
- No Lean 4 formal verification; 38 property-based tests validate invariants
- PDFA baseline uses simplified ALERGIA (tuned: 76–80% accuracy with 3–5 states)
- QCTL_F is a Pattinson–Schröder instantiation, not a novel logic
- Simple statistical baselines detect aggregate divergence comparably; CABER's advantage is structural
