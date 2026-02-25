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

# Real LLM experiments (requires OPENAI_API_KEY, ~$2)
source ~/.bashrc && python3 expanded_experiments.py
```

## Key Result: System Prompt Behavioral Divergence

Deployed against `gpt-4.1-nano` under 3 system-prompt configurations (270 API calls), CABER discovers that **7/15 prompts produce divergent behavior** depending on the system prompt:

| Configuration | States | Refusal % | Properties | Key Finding |
|---------------|--------|-----------|------------|-------------|
| Safety-strict | 2 | 48.9% | 3/3 pass | Robust refusal persistence |
| Creative-permissive | 2 | 33.3% | 2/3 pass | Refusal persistence **fails** (intended) |
| Instruction-rigid | 3 | 60.0% | 3/3 pass | Discovers emergent "terse" behavioral atom |

Bayesian posterior: P(divergence rate > 25%) = 97.3%, 95% HPD [25.0%, 69.9%].

**What CABER adds over simple statistics:** Chi-squared/MMD detect aggregate divergence comparably. CABER adds *structure*: reusable automata, graded satisfaction in [0,1], temporal multi-step reasoning, and Kantorovich drift metrics.

## Important Limitations

- **Not formal verification.** CABER is approximate behavioral testing inspired by coalgebraic methods. Do not rely on it for safety-critical certification.
- **Single model only**: all experiments use `gpt-4.1-nano` (non-frontier). Cross-model validation needed for generality.
- **PAC bounds vacuous** at operating sample sizes (54–90 samples vs ~143K required for ε=0.05). Use Bayesian posteriors for uncertainty quantification.
- **Classifier does not generalize**: leave-one-prompt-out CV yields 0.55–0.69 accuracy. Learned automaton is valid for queried prompts, not novel ones.
- **Non-functorial abstraction**: 0–27% inconsistency rate breaks coalgebraic morphism structure that underwrites the theoretical framework.
- **Paper proofs only**: no Lean 4/Coq mechanization. 38 property-based tests validate invariants computationally.
- **Safety fragment only**: liveness property verification remains open.
- **CoalCEGAR monotonicity violated** for 1/5 configurations due to statistical estimation noise.
- **Calibration error 0.28–0.73** for graded satisfaction scores.
- **PDFA baseline**: uses simplified ALERGIA, not full AALpy+PRISM (tuned PDFA achieves 76–80%).

## Paper

Technical paper: `implementation/tool_paper.tex` (compile with `pdflatex`).

Claim-to-evidence mapping: `grounding.json` (35 verified claims, 21 honest limitations documented).
