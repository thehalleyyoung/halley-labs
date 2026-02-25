# CABER: Coalgebraic Behavioral Auditing of Foundation Models

CABER audits black-box LLMs by learning finite-state behavioral automata and model-checking temporal specifications against them. Given only API access, it extracts a compact behavioral model, verifies multi-step safety properties (refusal persistence, sycophancy resistance, jailbreak resistance), and quantifies behavioral drift between model versions via Kantorovich bisimulation distance.

## Quickstart (30 seconds)

```bash
cd implementation

# Mock LLM validation — no API key needed, ~10s
python3 phase0_experiments.py

# Build Rust core + run 38 property-based tests
cargo build --release && cargo test -p caber-integration --test property_tests

# Real LLM experiments (requires OPENAI_API_KEY, ~$2)
source ~/.bashrc && python3 pathb_experiments.py
```

## What It Does

Deploy against `gpt-4.1-nano` under 3 system-prompt configurations (270 API calls):

| Configuration | States | Refusal % | Properties | Key Finding |
|---------------|--------|-----------|------------|-------------|
| Safety-strict | 2 | 46.7% | 3/3 pass | Robust refusal persistence |
| Creative-permissive | 2 | 33.3% | 2/3 pass | Refusal persistence **fails** (intended) |
| Instruction-rigid | 3 | 60.0% | 3/3 pass | Discovers emergent "terse" atom |

7/15 prompts diverge across configurations (Bayesian P(>25%) = 97.3%).

**CABER's advantage over simple statistics:** Chi-squared/MMD detect aggregate divergence comparably. CABER adds *structure*: reusable automata, graded satisfaction in [0,1], temporal multi-step reasoning, and Kantorovich drift metrics.

## Important Limitations

- **Single model only**: all experiments use `gpt-4.1-nano` (non-frontier). Cross-model validation needed for generality.
- **PAC bounds vacuous** at operating sample sizes (90 samples vs ~143K required for ε=0.05). Use Bayesian posteriors for uncertainty.
- **Classifier does not generalize**: leave-one-prompt-out CV yields 0.55–0.69 accuracy. Learned automaton is valid for queried prompts, not novel ones.
- **Non-functorial abstraction**: 0–27% inconsistency rate. Breaks coalgebraic morphism structure.
- **Paper proofs only**: no Lean 4/Coq mechanization. 38 property-based tests validate invariants.
- **Safety fragment only**: liveness property verification remains open.
- **PDFA baseline**: uses simplified ALERGIA, not full AALpy+PRISM (tuned PDFA achieves 76–80%).

## Paper

Technical paper: `implementation/tool_paper.tex` (compile with `pdflatex`).

Claim-to-evidence mapping: `grounding.json` (35 verified claims, 21 honest limitations documented).
