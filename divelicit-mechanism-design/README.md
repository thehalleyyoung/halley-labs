# DivFlow: Diverse LLM Response Selection via Sinkhorn-Guided Mechanism Design

Select maximally diverse, high-quality subsets from LLM response pools using optimal transport and mechanism design.

## 30-Second Quickstart

```bash
pip install numpy scipy z3-solver scikit-learn openai
cd implementation
python3 -m pytest tests/ -q  # 122 tests pass
```

```python
import numpy as np
from src.transport import sinkhorn_candidate_scores
from src.dpp import DPP
from src.kernels import RBFKernel

# Generate sample data: 100 LLM response embeddings + quality scores
embs = np.random.randn(100, 32)
quals = np.random.uniform(0.3, 1.0, 100)

# DivFlow: greedy Sinkhorn-guided selection (k=10)
selected = [int(np.argmax(quals))]
for _ in range(9):
    scores = sinkhorn_candidate_scores(embs, embs[selected], embs)
    combined = 0.7 * scores / max(abs(scores).max(), 1e-10) + 0.3 * quals
    combined[[s for s in selected]] = -np.inf
    selected.append(int(np.argmax(combined)))
print(f"DivFlow selected: {selected}")

# Compare to DPP baseline
dists = np.sqrt(np.sum((embs[:, None] - embs[None, :]) ** 2, axis=-1))
bw = float(np.median(dists[dists > 0]))
K = RBFKernel(bandwidth=bw).gram_matrix(embs) + 1e-6 * np.eye(100)
dpp_selected = DPP(K).greedy_map(10)
print(f"DPP selected:     {dpp_selected}")
```

## Run Experiments

```bash
cd implementation

# Full synthetic evaluation (20 prompts × 200 responses, IC, Z3, composition)
python3 experiments/run_scaled_evaluation.py

# LLM evaluation (requires OPENAI_API_KEY)
source ~/.bashrc
python3 experiments/run_llm_evaluation.py

# Tests (122 passing)
python3 -m pytest tests/ -q
```

## Key Results

| Claim | Result | Source |
|-------|--------|--------|
| Quasi-linearity | Exact (error 2.57e-16) | `algebraic_proof.py` |
| IC violation rate | 26.5% (1,200 tests), 100% Type A | `ic_analysis.py` |
| ε-IC bound | Corrected: 7.92 (validated, max gain 0.56) | `composition_theorem.py` |
| Z3 grid-certified | 2/8 agents (25%) at grid=15 | `z3_verification.py` |
| Z3 soundness gap | L=13.19, L·h=0.94 | `z3_verification.py` |
| **LLM: DivFlow** | **68.1% coverage, 0.698 quality** | `run_llm_evaluation.py` |
| LLM: DPP | 86.2% coverage, 0.546 quality | `run_llm_evaluation.py` |
| LLM: TopQuality | 43.8% coverage, 0.872 quality | `run_llm_evaluation.py` |
| Synthetic: DivFlow vs TopQ | +14.0% coverage (d=2.22, p<0.0001) | `run_scaled_evaluation.py` |
| DPP ≠ TopQuality | ✓ (diversity-only kernel, no quality weighting) | `run_scaled_evaluation.py` |

**DivFlow achieves the Pareto-optimal quality–diversity tradeoff**: higher coverage than TopQuality, higher quality than DPP.

## Architecture

```
src/
├── transport.py            Sinkhorn divergence, dual potentials, candidate scoring
├── algebraic_proof.py      Non-tautological proof of quasi-linearity via welfare pipeline
├── composition_theorem.py  Corrected ε-IC bounds for approximately submodular welfare
├── ic_analysis.py          IC violation root-cause analysis, VCG condition checking
├── z3_verification.py      Z3 SMT with grid-certified vs soundness-certified distinction
├── mechanism.py            DivFlow, VCG, DPP, MMR, k-Medoids selection
├── dpp.py                  DPP greedy MAP (diversity-only L-kernel)
├── sensitivity_analysis.py Hyperparameter sensitivity (ε, λ, k sweeps)
├── coverage.py             ε-net certificates, Clopper-Pearson/Wilson CIs, bootstrap
├── scoring_rules.py        Proper scoring rules (log, Brier, spherical, CRPS, energy)
├── kernels.py              RBF, adaptive, multi-scale kernels
└── diversity_metrics.py    Cosine diversity, log-det, Vendi score
```

## Paper

```bash
cd theory && pdflatex paper.tex  # 13 pages main + appendix
```
