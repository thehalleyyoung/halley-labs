# CSAT: Causal-Shielded Adaptive Trading

Regime-aware causal discovery + posterior-predictive safety shields with PAC-Bayes certificates for adaptive trading strategies.

## 30-Second Quickstart

```bash
pip install numpy scipy networkx scikit-learn matplotlib
cd implementation/
```

```python
import numpy as np
from causal_trading.regime import StickyHDPHMM
from causal_trading.shield import PosteriorPredictiveShield, BoundedDrawdownSpec
from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer

# 1. Regime detection — Sticky HDP-HMM finds 3 regimes in synthetic data
rng = np.random.default_rng(42)
data = np.concatenate([rng.normal(0, 1, (200, 5)), rng.normal(2, 1.5, (200, 5))])
hmm = StickyHDPHMM(K_max=5, n_iter=50, random_state=42)
hmm.fit(data)
print("Regimes found:", len(np.unique(hmm.states_)))

# 2. Shield synthesis — restrict actions via drawdown safety spec
shield = PosteriorPredictiveShield(n_states=10, n_actions=5, delta=0.05)
shield.add_spec(BoundedDrawdownSpec(max_drawdown=0.10), name="drawdown")
shield.synthesize()
print("Permitted actions (state 0):", shield.get_permitted_actions(state=0))

# 3. PAC-Bayes vacuity analysis — non-vacuous bounds (< 0.5) at all sample sizes
analyzer = PACBayesVacuityAnalyzer(n_abstract_states_per_regime=10, n_actions=5)
bounds = analyzer.compute_bound_curve(K=3, n_values=np.array([100, 1000, 10000]))
for n, b in zip([100, 1000, 10000], bounds):
    print(f"  n={n:>5d}  bound={b:.6f}  vacuous={b >= 0.5}")
```

**Output:**
```
Regimes found: 3
Permitted actions (state 0): [False False False False  True]
  n=  100  bound=0.267162  vacuous=False
  n= 1000  bound=0.288969  vacuous=False
  n=10000  bound=0.108086  vacuous=False
```

All three bounds are non-vacuous (< 0.5), meaning the PAC-Bayes certificate provides meaningful safety guarantees even with only 100 samples.

## Architecture

```
Market Data (T×D array)
    │
    ▼
┌──────────────┐     ┌──────────────────────────────────┐
│ HSIC-Lasso   │────▶│ Coupled Inference (EM)            │
│ Feature Sel. │     │  E-step: Sticky HDP-HMM           │
└──────────────┘     │  M-step: PC + HSIC per regime      │
                     │  + SpuriousFixedPointDetector       │
                     └──────────┬───────────────────────┘
                                │
                  ┌─────────────┴──────────────┐
                  ▼                            ▼
         ┌────────────────┐          ┌─────────────────┐
         │ SCIT Invariance│          │ Per-regime DAGs  │
         │ (e-values)     │          │ + Student-t emit │
         └───────┬────────┘          └────────┬────────┘
                 └─────────────┬──────────────┘
                               ▼
          ┌──────────────────────────────────┐
          │ Shield Synthesis                 │
          │  Bounded LTL + liveness specs    │
          │  PAC-Bayes certificate            │
          │  Adaptive δ + graceful degrade   │
          │  State abstraction verification   │
          └──────────────┬───────────────────┘
                         ▼
          ┌──────────────────────────────────┐
          │ Shielded Portfolio Optimisation  │
          │  + Error decomposition            │
          │  + Independent verification       │
          └──────────────────────────────────┘
```

## Key Capabilities

| Capability | Description |
|---|---|
| **Regime-Indexed SCMs** | Joint Bayesian inference over latent regimes and per-regime causal DAGs via EM alternation (Sticky HDP-HMM + PC-HSIC) |
| **Posterior-predictive shields** | Restrict actions to those satisfying bounded LTL safety specs with PAC-Bayes certified P ≥ 1 − δ |
| **State abstraction soundness** | Conservative overapproximation with monotone refinement (gap 0.024 → 0 as grid refines) |
| **4-stage error decomposition** | Per-stage error budgets (regime, DAG, invariance, shield) replacing monolithic ε₁ + ε₂ |
| **Bounded liveness specs** | DrawdownRecovery, LossRecovery, PositionReduction, RegimeTransition: G(trigger → F[0,H] recovery) |
| **Student-t emissions** | Heavy-tailed regime model (BIC 5552 vs Gaussian 6131) |
| **Spurious fixed-point detection** | EM convergence monitoring with contraction rate and Lyapunov analysis |
| **Adaptive delta** | Shield δ scales with posterior concentration for permissivity tuning |
| **Graceful degradation** | Always permits at least one (safest) action when no action meets threshold |
| **Multi-instrument evaluation** | Cross-asset-class experiments (equity, FX, crypto) with ARI/SHD/PAC-Bayes metrics |
| **Sensitivity analysis** | Systematic hyperparameter sweeps showing δ is the only sensitive parameter |
| **Independent verification** | Standalone certificate re-derivation using only NumPy/SciPy |
| **HSIC-Lasso** | Kernel-based feature selection for nonlinear causal relationships (3/3 vs LASSO 1/3) |

## CLI Highlights

```bash
# Generate synthetic regime-switching market data
causal-trading generate-data --n-features 20 --n-regimes 3 --T 5000 -o data/

# Train coupled regime-causal model
causal-trading train --data data/features.csv --config config.json

# Run shielded backtest with walk-forward validation
causal-trading backtest --model model.pkl --data data/ --output results/

# Generate formal safety certificate
causal-trading certificate --model model.pkl -o certificate.json

# Verify 7 safety properties (PAC-Bayes, composition, soundness, ...)
causal-trading shield-check --model model.pkl --delta 0.05

# Live monitoring dashboard (regime, shield, anomaly tracking)
causal-trading monitor --model model.pkl

# Component ablation studies
causal-trading ablation --data data/ --output ablation_results/
```

## Reproducing Paper Results

All experiments are deterministic (seed 42) and run on a single CPU.

```bash
# Run all experiments (~9 minutes total)
python3 -m experiments.run_all_experiments
python3 -m experiments.run_sensitivity
python3 -m experiments.run_multi_instrument

# Results written to experiments/results/
#   state_abstraction.json        — Table 3 (overapproximation gaps)
#   error_decomposition.json      — Table 4 (per-stage errors)
#   bounded_liveness.json         — Table 5 (liveness satisfaction rates)
#   emission_comparison.json      — Table 6 (Gaussian vs Student-t)
#   pac_bayes_vacuity.json        — Table 2 (bound vs sample size)
#   independent_verification.json — Table 8 (audit discrepancies)
#   sensitivity/sensitivity_report.json — Hyperparameter sensitivity
#   multi_instrument.json         — Cross-asset-class evaluation
```

## Project Structure

```
causal_trading/
├── regime/          StickyHDPHMM, StudentTEmission, EmissionModelSelector
├── causal/          PCAlgorithm, HSIC, AdditiveNoiseModel
├── invariance/      SCITAlgorithm, WealthProcess, e-values
├── coupled/         CoupledInference, ConvergenceAnalyzer, SpuriousFixedPointDetector
├── shield/          PosteriorPredictiveShield, BoundedLivenessSpec, PACBayes
├── verification/    StateAbstraction, IntervalArithmetic, IndependentVerifier
├── proofs/          DecomposedCompositionTheorem, Certificate
├── evaluation/      ErrorDecomposition, SensitivityAnalyzer
├── portfolio/       ShieldedMeanVarianceOptimizer, CausalFeatureSelector
├── market/          SyntheticMarketGenerator
└── monitoring/      ShieldMonitor, PermissivityTracker

experiments/
├── run_all_experiments.py     Core paper experiments
├── run_sensitivity.py         Hyperparameter sensitivity sweeps
└── run_multi_instrument.py    Multi-asset-class evaluation
```

## Requirements

- Python ≥ 3.9
- NumPy ≥ 1.24, SciPy ≥ 1.10, NetworkX ≥ 3.0, scikit-learn ≥ 1.2, Matplotlib ≥ 3.7

## License

MIT
