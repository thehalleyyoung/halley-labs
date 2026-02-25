# CABER: Coalgebraic Behavioral Auditing of Foundation Models

A verification engine that treats black-box LLMs as coalgebras, extracts finite
behavioral automata via active learning (PCL\*), and model-checks temporal
specifications (QCTL_F) with PAC-style soundness guarantees.

## 30-Second Quickstart

```bash
# Run real LLM experiment + PDFA baseline + state scaling (requires OPENAI_API_KEY)
source ~/.bashrc && python3 phase_b_experiments.py

# Run Phase 0 mock LLM validation
python3 phase0_experiments.py

# Build Rust core and run 38 property-based tests
cargo build --release
cargo test -p caber-integration --test property_tests
```

### Headline Result — Real LLM Validation

```
$ python3 phase_b_experiments.py
EXPERIMENT 1: Real LLM Validation (gpt-4.1-nano)
  Behavioral automaton: 3 states (initial, compliant, refusal)
  Behavior distribution: compliant 76.7%, refusal 23.3%
  Refusal persistence:   PASS (3/3 trials)
  Sycophancy resistance: PASS (3/3 trials)
  Jailbreak resistance:  PASS (3/3 trials)
  Total API calls: 63

EXPERIMENT 2: PDFA+PRISM Baseline Comparison
  CABER avg: 30 states, 90.1% accuracy
  PDFA  avg: 293 states, 68.5% accuracy
  → CABER learns 5-10x more compact automata

EXPERIMENT 3: State Scaling (3 → 100 ground-truth states)
  PCL* maintains ≥96% accuracy up to 100 states
  Query complexity: O(n^1.42)
```

## Key Results

| Experiment | Result |
|------------|--------|
| Real LLM (gpt-4.1-nano) | 3-state automaton, 3/3 properties pass, 63 API calls |
| CABER vs PDFA baseline | 5-10x more compact automata, +21.6pp accuracy |
| State scaling (3-100) | ≥96% accuracy, O(n^1.42) query scaling |
| Mock LLM validation | ≤40 states, ≥92% accuracy across 4 configurations |
| Classifier robustness | ≥99.2% verdict accuracy at 20% error rate |

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  7. Evaluation Layer   — Python harness, baselines, drift simulation │
├──────────────────────────────────────────────────────────────────────┤
│  6. Certificate Layer  — generation, verification, audit reports     │
├──────────────────────────────────────────────────────────────────────┤
│  5. Bisimulation Layer — exact/quantitative, Kantorovich lifting     │
├──────────────────────────────────────────────────────────────────────┤
│  4. Verification Layer — QCTL_F model checking, fixed-point engine   │
├──────────────────────────────────────────────────────────────────────┤
│  3. Abstraction Layer  — CoalCEGAR loop, lattice traversal           │
├──────────────────────────────────────────────────────────────────────┤
│  2. Learning Layer     — PCL* algorithm, observation tables          │
├──────────────────────────────────────────────────────────────────────┤
│  1. Query Layer        — black-box model interface, scheduling       │
└──────────────────────────────────────────────────────────────────────┘
```

## Specification Templates

Six built-in templates compile to QCTL_F formulas:

| Template | What It Checks |
|----------|----------------|
| `RefusalPersistence` | Once refused, model refuses all paraphrases |
| `ParaphraseInvariance` | Equivalent prompts → similar outputs |
| `VersionStability` | Behavioral distance between versions ≤ δ |
| `SycophancyResistance` | Opinion reversal rate ≤ p after disagreement |
| `InstructionHierarchy` | System-prompt dominates user-prompt |
| `JailbreakResistance` | Refusal probability ≥ p after N priming turns |

## Project Structure

```
caber-core/         Rust — core engine
  coalgebra/        types, semiring, distributions, functors, bandwidth
  learning/         PCL* algorithm, observation table, query oracle
  abstraction/      CoalCEGAR, lattice, Galois connections, refinement
  temporal/         QCTL_F syntax, semantics, spec templates
  model_checker/    fixed-point engine, graded satisfaction, witness gen
  bisimulation/     exact/quantitative bisimulation, Kantorovich lifting
  certificate/      audit report generation, verification
  query/            black-box model interface, scheduler, consistency
caber-cli/          command-line interface
caber-examples/     runnable demos (refusal_audit, phase0, etc.)
caber-integration/  38 property-based + end-to-end tests
caber-python/       mock LLMs, baselines, evaluation harness
phase0_experiments.py           Phase 0 validation script
classifier_robustness_analysis.py   Robustness Monte Carlo analysis
tool_paper.tex                  Paper (arxiv format)
```

## Prerequisites

- **Rust** 1.75+ (2021 edition)
- **Python** 3.10+ (for evaluation harness)

## Build & Test

```bash
cargo build --release
cargo test                                          # all tests
cargo test -p caber-integration --test property_tests  # 38 property tests
```

## Experiments

```bash
python3 phase_b_experiments.py                 # → phase_b_results.json (real LLM + baselines + scaling)
python3 phase0_experiments.py                  # → phase0_results.json (mock LLM validation)
python3 classifier_robustness_analysis.py      # → classifier_robustness_results.json
```

## Limitations

- Real LLM experiment uses **gpt-4.1-nano** (small model); frontier model validation is future work
- Alphabet abstraction (embedding clustering) is **non-functorial**; adversarial robustness unvalidated
- No Lean 4 formalization (38 property-based tests bridge the gap)
- PDFA baseline uses simplified ALERGIA, not full AALpy+PRISM
- PAC bounds are conservative due to causal coupling between pipeline stages
- QCTL_F is an instantiation of Pattinson–Schröder framework, not a novel logic
