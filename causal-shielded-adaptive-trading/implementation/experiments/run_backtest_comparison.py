#!/usr/bin/env python3
"""
Comprehensive walk-forward backtest comparing CSAT against baselines.

Generates realistic regime-switching market data (fat tails, GARCH volatility
clustering, causal structure) and evaluates strategies across multiple asset
classes, crisis scenarios, and market conditions.

Usage
-----
    cd implementation/
    python3 experiments/run_backtest_comparison.py
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

_IMPL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _IMPL_DIR not in sys.path:
    sys.path.insert(0, _IMPL_DIR)

from causal_trading.market.synthetic import SyntheticMarketGenerator
from causal_trading.evaluation.baseline_strategies import (
    BuyAndHoldStrategy,
    MomentumStrategy,
    MeanReversionStrategy,
    RiskParityStrategy,
    UnshieldedMeanVarianceStrategy,
    OracleStrategy,
    CSATStrategy,
    compare_strategies,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_backtest_comparison")

RESULTS_DIR = Path(__file__).parent / "results"
SEED = 42


def _save_json(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info("Saved %s", path)


# ---------------------------------------------------------------------------
# Scenario definitions
# ---------------------------------------------------------------------------

SCENARIOS = {
    "equity_bull_bear": {
        "description": "Equity-like: 3 regimes (bull/bear/crash) with GARCH vol",
        "n_features": 8,
        "n_regimes": 3,
        "T": 2000,
        "regime_persistence": 0.97,
        "snr": 1.5,
        "fat_tail_df": 5.0,
        "use_garch": True,
        "regime_actions": {0: 2, 1: -1, 2: -3},  # bull/bear/crash
    },
    "fx_trending": {
        "description": "FX-like: 2 regimes (trending/ranging) with high persistence",
        "n_features": 6,
        "n_regimes": 2,
        "T": 2000,
        "regime_persistence": 0.98,
        "snr": 1.0,
        "fat_tail_df": 8.0,
        "use_garch": True,
        "regime_actions": {0: 2, 1: 0},
    },
    "crypto_volatile": {
        "description": "Crypto-like: 4 regimes with extreme volatility",
        "n_features": 5,
        "n_regimes": 4,
        "T": 2000,
        "regime_persistence": 0.93,
        "snr": 0.8,
        "fat_tail_df": 3.0,
        "use_garch": True,
        "regime_actions": {0: 3, 1: 1, 2: -2, 3: -3},
    },
    "crisis_flash_crash": {
        "description": "Flash crash: sudden regime break at T=1000",
        "n_features": 8,
        "n_regimes": 3,
        "T": 2000,
        "regime_persistence": 0.99,
        "snr": 2.0,
        "fat_tail_df": 4.0,
        "use_garch": True,
        "regime_actions": {0: 2, 1: 0, 2: -3},
    },
    "low_snr": {
        "description": "Low SNR: noisy regime with weak causal signals",
        "n_features": 10,
        "n_regimes": 3,
        "T": 2000,
        "regime_persistence": 0.95,
        "snr": 0.3,
        "fat_tail_df": 6.0,
        "use_garch": True,
        "regime_actions": {0: 1, 1: 0, 2: -1},
    },
}


def _inject_flash_crash(
    returns: np.ndarray,
    features: np.ndarray,
    crash_start: int = 1000,
    crash_duration: int = 10,
    crash_magnitude: float = -0.15,
    rng: np.random.Generator = None,
) -> tuple:
    """Inject a flash crash into synthetic data."""
    rng = rng or np.random.default_rng(42)
    r = returns.copy()
    f = features.copy()

    # Sharp drawdown
    for t in range(crash_start, min(crash_start + crash_duration, len(r))):
        r[t] = crash_magnitude / crash_duration + rng.normal(0, 0.02)
        f[t] *= rng.uniform(0.5, 1.5, size=f.shape[1])

    # Elevated volatility in aftermath
    aftermath = min(crash_start + crash_duration + 50, len(r))
    for t in range(crash_start + crash_duration, aftermath):
        r[t] *= rng.uniform(1.5, 3.0)

    return r, f


def run_scenario(
    name: str,
    config: Dict[str, Any],
    seed: int = SEED,
) -> Dict[str, Any]:
    """Run one backtest scenario."""
    logger.info("Running scenario: %s", name)
    t0 = time.time()

    gen = SyntheticMarketGenerator(
        n_features=config["n_features"],
        n_regimes=config["n_regimes"],
        regime_persistence=config["regime_persistence"],
        snr=config.get("snr", 1.0),
        fat_tail_df=config.get("fat_tail_df", 5.0),
        use_garch=config.get("use_garch", True),
        seed=seed,
    )
    dataset = gen.generate(T=config["T"])
    features = dataset.features
    returns = dataset.returns

    # Inject crisis for flash crash scenario
    if name == "crisis_flash_crash":
        rng = np.random.default_rng(seed)
        returns, features = _inject_flash_crash(returns, features, rng=rng)

    regime_actions = config.get("regime_actions", {0: 1, 1: 0, 2: -1})

    strategies = [
        BuyAndHoldStrategy(action=1),
        MomentumStrategy(lookback=20),
        MeanReversionStrategy(lookback=50),
        RiskParityStrategy(lookback=60),
        UnshieldedMeanVarianceStrategy(lookback=60),
        OracleStrategy(
            regime_labels=dataset.ground_truth.regime_labels,
            regime_actions=regime_actions,
        ),
        CSATStrategy(
            n_regimes=config["n_regimes"],
            n_features=config["n_features"],
            seed=seed,
        ),
    ]

    results = compare_strategies(
        features, returns, strategies, warmup=100, cost_bps=5.0,
    )

    elapsed = time.time() - t0
    logger.info("Scenario %s completed in %.1fs", name, elapsed)

    scenario_result = {
        "scenario": name,
        "description": config["description"],
        "T": config["T"],
        "n_features": config["n_features"],
        "n_regimes": config["n_regimes"],
        "elapsed_s": round(elapsed, 2),
        "strategies": {},
    }

    for sname, sr in results.items():
        scenario_result["strategies"][sname] = {
            "total_return": round(sr.total_return, 6),
            "annualised_return": round(sr.annualised_return, 6),
            "annualised_vol": round(sr.annualised_vol, 6),
            "sharpe_ratio": round(sr.sharpe_ratio, 4),
            "sortino_ratio": round(sr.sortino_ratio, 4),
            "max_drawdown": round(sr.max_drawdown, 6),
            "calmar_ratio": round(sr.calmar_ratio, 4),
            "n_trades": sr.n_trades,
        }

    return scenario_result


# ---------------------------------------------------------------------------
# Crisis robustness analysis
# ---------------------------------------------------------------------------

def run_crisis_robustness(seed: int = SEED) -> Dict[str, Any]:
    """Test CSAT behaviour under escalating crisis severity."""
    logger.info("Running crisis robustness analysis")
    t0 = time.time()

    gen = SyntheticMarketGenerator(
        n_features=8, n_regimes=3, regime_persistence=0.97,
        snr=1.5, fat_tail_df=5.0, use_garch=True, seed=seed,
    )
    dataset = gen.generate(T=2000)
    features = dataset.features
    base_returns = dataset.returns

    magnitudes = [-0.05, -0.10, -0.15, -0.20, -0.30]
    results = []
    rng = np.random.default_rng(seed)

    for mag in magnitudes:
        returns, feat = _inject_flash_crash(
            base_returns, features,
            crash_magnitude=mag,
            crash_start=1000,
            crash_duration=5,
            rng=np.random.default_rng(seed),
        )

        csat = CSATStrategy(n_regimes=3, n_features=8, seed=seed)
        buy_hold = BuyAndHoldStrategy(action=1)

        from causal_trading.evaluation.baseline_strategies import run_walk_forward_backtest
        csat_result = run_walk_forward_backtest(feat, returns, csat, warmup=100)
        bh_result = run_walk_forward_backtest(feat, returns, buy_hold, warmup=100)

        # Measure crisis-period behavior
        crisis_start = 1000 - 100  # adjusted for warmup
        crisis_end = min(crisis_start + 50, len(csat_result.actions))
        if crisis_start >= 0 and crisis_end > crisis_start:
            crisis_actions = csat_result.actions[crisis_start:crisis_end]
            avg_position = float(np.mean(crisis_actions))
            min_position = int(np.min(crisis_actions))
            position_reduction_time = int(np.argmin(crisis_actions))
        else:
            avg_position = 0.0
            min_position = 0
            position_reduction_time = 0

        results.append({
            "crash_magnitude": mag,
            "csat_max_dd": round(csat_result.max_drawdown, 6),
            "bh_max_dd": round(bh_result.max_drawdown, 6),
            "dd_reduction": round(
                1 - csat_result.max_drawdown / max(bh_result.max_drawdown, 1e-10), 4
            ),
            "csat_sharpe": round(csat_result.sharpe_ratio, 4),
            "bh_sharpe": round(bh_result.sharpe_ratio, 4),
            "avg_crisis_position": round(avg_position, 2),
            "min_crisis_position": min_position,
            "position_reduction_time_steps": position_reduction_time,
        })

    elapsed = time.time() - t0
    return {
        "analysis": "crisis_robustness",
        "crash_magnitudes": magnitudes,
        "results": results,
        "elapsed_s": round(elapsed, 2),
    }


# ---------------------------------------------------------------------------
# Shield permissivity analysis under uncertainty
# ---------------------------------------------------------------------------

def run_shield_permissivity_analysis(seed: int = SEED) -> Dict[str, Any]:
    """Analyze shield permissivity across different uncertainty levels."""
    from causal_trading.shield.shield_synthesis import PosteriorPredictiveShield
    from causal_trading.shield.bounded_liveness_specs import DrawdownRecoverySpec
    from causal_trading.shield.pac_bayes import PACBayesVacuityAnalyzer

    logger.info("Running shield permissivity analysis")

    deltas = [0.01, 0.02, 0.05, 0.10, 0.20, 0.50]
    results = []

    for delta in deltas:
        shield = PosteriorPredictiveShield(
            n_states=10, n_actions=7, delta=delta,
        )
        shield.add_spec(
            DrawdownRecoverySpec(threshold=0.10, horizon=20),
            name="drawdown",
        )
        shield.synthesize()

        total_permitted = 0
        total_pairs = 0
        for s in range(10):
            permitted = shield.get_permitted_actions(state=s)
            total_permitted += int(np.sum(permitted))
            total_pairs += len(permitted)

        permissivity = total_permitted / max(total_pairs, 1)
        results.append({
            "delta": delta,
            "permissivity": round(permissivity, 4),
            "total_permitted": total_permitted,
            "total_pairs": total_pairs,
        })

    # PAC-Bayes bound vs delta
    analyzer = PACBayesVacuityAnalyzer(
        n_abstract_states_per_regime=10, n_actions=7,
    )
    n_values = np.array([100, 500, 1000, 5000, 10000])
    pac_bayes_results = []
    for n in n_values:
        bounds = analyzer.compute_bound_curve(K=3, n_values=np.array([n]))
        pac_bayes_results.append({
            "n": int(n),
            "bound": round(float(bounds[0]), 6),
            "vacuous": bool(bounds[0] >= 0.5),
        })

    return {
        "analysis": "shield_permissivity",
        "delta_sweep": results,
        "pac_bayes_bounds": pac_bayes_results,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    all_results = {
        "experiment": "backtest_comparison",
        "seed": SEED,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # Run all scenarios
    scenario_results = {}
    for name, config in SCENARIOS.items():
        try:
            scenario_results[name] = run_scenario(name, config, seed=SEED)
        except Exception as e:
            logger.error("Scenario %s failed: %s", name, e)
            scenario_results[name] = {"error": str(e)}
    all_results["scenarios"] = scenario_results

    # Crisis robustness
    try:
        all_results["crisis_robustness"] = run_crisis_robustness(seed=SEED)
    except Exception as e:
        logger.error("Crisis robustness failed: %s", e)
        all_results["crisis_robustness"] = {"error": str(e)}

    # Shield permissivity
    try:
        all_results["shield_permissivity"] = run_shield_permissivity_analysis(seed=SEED)
    except Exception as e:
        logger.error("Shield permissivity failed: %s", e)
        all_results["shield_permissivity"] = {"error": str(e)}

    # Summary table
    summary_table = []
    for sname, sresult in scenario_results.items():
        if "error" in sresult:
            continue
        row = {"scenario": sname}
        for strat_name, strat_data in sresult.get("strategies", {}).items():
            row[f"{strat_name}_sharpe"] = strat_data.get("sharpe_ratio", None)
            row[f"{strat_name}_max_dd"] = strat_data.get("max_drawdown", None)
        summary_table.append(row)
    all_results["summary_table"] = summary_table

    _save_json(all_results, RESULTS_DIR / "backtest_comparison.json")

    # Print summary
    print("\n" + "=" * 80)
    print("BACKTEST COMPARISON RESULTS")
    print("=" * 80)
    for sname, sresult in scenario_results.items():
        if "error" in sresult:
            print(f"\n{sname}: ERROR - {sresult['error']}")
            continue
        print(f"\n{sname} ({sresult.get('description', '')})")
        print("-" * 60)
        print(f"{'Strategy':<20} {'Sharpe':>8} {'MaxDD':>8} {'Return':>10}")
        print("-" * 60)
        for strat_name, strat_data in sresult.get("strategies", {}).items():
            print(
                f"{strat_name:<20} "
                f"{strat_data.get('sharpe_ratio', 0):>8.3f} "
                f"{strat_data.get('max_drawdown', 0):>8.3%} "
                f"{strat_data.get('total_return', 0):>10.3%}"
            )
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
