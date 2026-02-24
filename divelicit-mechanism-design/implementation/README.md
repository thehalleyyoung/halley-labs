# DivFlow: Diverse LLM Response Selection via Sinkhorn-Guided Mechanism Design

## 30-Second Quickstart

```bash
pip install numpy scipy openai
cd implementation
python3 -m pytest tests/ -q  # 78 tests pass

# Run the full evaluation (uses cached LLM embeddings)
python3 experiments/run_scaled_experiment.py
```

```python
from src.transport import sinkhorn_candidate_scores
from src.coverage import estimate_coverage, clopper_pearson_ci
from src.composition_theorem import verify_ic_with_ci
import numpy as np

# Select diverse subset from embeddings
embs = np.random.randn(100, 64)  # your embeddings
quals = np.random.rand(100)       # quality scores
history = embs[:3]                # already selected
scores = sinkhorn_candidate_scores(embs[3:], history, embs)
best = 3 + int(np.argmax(scores))

# Coverage certificate with CI
cert = estimate_coverage(embs[:10], epsilon=0.5)
print(f"Coverage: {cert.coverage_fraction:.3f} [{cert.ci_lower:.3f}, {cert.ci_upper:.3f}]")
```

## What This Does

DivFlow selects diverse, high-quality subsets from LLM response pools using **Sinkhorn divergence** (optimal transport) to measure coverage and **VCG mechanism design** for incentive compatibility.

**Key results** (500 GPT-4.1-nano responses, 25 topics, k=10):
- **DivFlow**: 40% topic coverage, quality 0.904 [0.880, 0.913]
- **DPP**: 8% topic coverage (5× worse)
- **VCG-DivFlow**: 7.0% IC violation rate [2.9%, 13.9%]
- **Scoring rules**: 0/500 properness violations × 4 rules

## Architecture

```
src/
├── transport.py           # Sinkhorn divergence, dual potentials, candidate scoring
├── mechanism.py           # VCG, DivFlow, DPP, MMR, k-Medoids selection
├── coverage.py            # Coverage certificates, Clopper-Pearson/Wilson CIs, bootstrap
├── composition_theorem.py # ε-IC verification, composition theorem, adversarial testing
├── kernels.py             # RBF, Matern, adaptive, multi-scale, manifold-adaptive
├── scoring_rules.py       # Proper scoring rules + energy-augmented variant
├── dpp.py                 # DPP greedy MAP
├── diversity_metrics.py   # Cosine diversity, log-det, coverage metrics
└── agents.py              # Simulated LLM agents
```

## Running Tests

```bash
python3 -m pytest tests/ -q   # 78 tests, ~4 seconds
```

## Running Experiments

```bash
# Full scaled experiment (8 experiments, ~45 min with cached embeddings)
python3 experiments/run_scaled_experiment.py

# Results saved to experiments/scaled_experiment_results.json
```

## Paper

The paper is in `theory/paper.tex`. Build with:
```bash
cd theory && pdflatex paper.tex && pdflatex paper.tex
```
