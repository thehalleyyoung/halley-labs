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

# Scaled experiments (requires OPENAI_API_KEY, ~1552 API calls, ~$3)
source ~/.bashrc && python3 scaled_experiments.py

# Stability analysis (no API key needed, uses cached embeddings)
python3 -c "from caber.classifiers.stable_abstraction import *; print(compute_abstraction_gap.__doc__)"
```

## Key Results (1,552 API calls on gpt-4.1-nano)

### 1. Scaled Behavioral Automata (4 configurations, 50 prompt types)

| Configuration | States | Prompts | Calls | Dominant Atom |
|---|---|---|---|---|
| safety_strict | 2 | 50 | 250 | compliant (62%), refusal (38%) |
| creative_permissive | 4 | 50 | 250 | compliant (86%), refusal (12%) |
| instruction_rigid | 3 | 50 | 250 | compliant (72%), refusal (18%) |
| balanced_helpful | 4 | 50 | 250 | compliant (78%), refusal (18%) |

Plus 552 multi-turn conversation calls across 8 scenarios × 3 trials × 4 configurations.

### 2. Temporal Advantage over Chi-Squared

In **21/48 (44%) multi-turn configuration pairs**, CABER's transition-based analysis detects temporal behavioral patterns that chi-squared marginal tests miss. Examples: escalation timing, behavioral drift under pushback, entropy rate divergence.

### 3. Stability-Constrained Abstraction (non-functoriality fix)

The `StableAbstractionLayer` reduces abstraction inconsistency from 3.5–7% (boundary cases) to <0.2% via:
- **Margin-based rejection**: responses near cluster boundaries flagged for stabilization
- **Majority-vote stabilization** (K=11): eliminates flip errors for ambiguous inputs
- **Provable guarantee**: 85% of embeddings have margin ≥0.10 (provably stable under perturbation)
- **Effective error**: ε_abs^eff ≤ 2.25×10⁻⁴ (1,200× improvement over raw abstraction)

### 4. Compositional Specifications

CABER evaluates conjunctive properties (safety ∧ helpfulness) and temporal properties (consistency under pushback) inexpressible as single frequency tests.

## Important Limitations

- **Not formal verification.** CABER is approximate behavioral testing. Do not rely on it for safety-critical certification.
- **Single model only**: all experiments use `gpt-4.1-nano` (non-frontier). Cross-model validation needed.
- **PAC bounds vacuous** at operating sample sizes (250 per config vs ~143K required for ε=0.05). Bayesian posteriors recommended.
- **Non-functorial abstraction mitigated but not eliminated**: stability layer achieves 85% provably stable, but 15% require majority-vote fallback.
- **Paper proofs only**: no Lean 4/Coq mechanization. 38 property-based tests validate invariants.
- **Bisimulation distances**: trivially zero in current automaton construction (deterministic output distributions per state); transition matrix distances provide the meaningful comparison.

## Paper

Technical paper: `implementation/tool_paper.tex` (compile with `pdflatex`).

Claim-to-evidence mapping: `grounding.json`.
