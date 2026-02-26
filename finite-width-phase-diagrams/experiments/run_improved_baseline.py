#!/usr/bin/env python3
"""
Improved baseline comparison with corrected PhaseKit initialization.

Re-runs the 6-method baseline comparison with the improved PhaseKit init
that uses the exact edge-of-chaos sigma_w* with depth-aware corrections.
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from baselines import (
    xavier_init, kaiming_init, phasekit_init, lsuv_init,
    train_mlp, apply_activation, activation_derivative,
)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def run_improved_baseline():
    """Run improved baseline comparison."""
    print("=" * 60)
    print("IMPROVED BASELINE COMPARISON")
    print("=" * 60)

    depths = [3, 5, 10, 20]
    widths = [64, 128]
    activations = ["relu", "tanh", "gelu", "silu"]
    seeds = [42, 123]
    n_steps = 300
    lr = 0.01

    results = {}
    method_wins = {"xavier": 0, "kaiming": 0, "phasekit": 0, "lsuv": 0}
    n_valid = 0

    for depth in depths:
        for width in widths:
            for act in activations:
                config_key = f"D{depth}_W{width}_{act}"
                dims = [10] + [width] * depth + [1]

                # Generate data
                rng = np.random.RandomState(42)
                X_train = rng.randn(500, 10)
                y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] + \
                          0.3 * np.cos(X_train[:, 2]) + rng.randn(500) * 0.1
                X_test = rng.randn(100, 10)
                y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1] + \
                         0.3 * np.cos(X_test[:, 2])

                config_results = {}
                for method_name, init_fn in [
                    ("xavier", lambda s: xavier_init(dims, seed=s)),
                    ("kaiming", lambda s: kaiming_init(dims, act, seed=s)),
                    ("phasekit", lambda s: phasekit_init(dims, act, seed=s)),
                    ("lsuv", lambda s: lsuv_init(dims, act, X_calib=X_train[:256], seed=s)),
                ]:
                    losses = []
                    for seed in seeds:
                        init = init_fn(seed)
                        res = train_mlp(
                            init.weights, init.biases,
                            X_train, y_train, X_test, y_test,
                            activation=act, n_steps=n_steps, lr=lr
                        )
                        losses.append(res["final_loss"])

                    mean_loss = np.mean(losses)
                    config_results[method_name] = {
                        "mean_loss": float(mean_loss),
                        "losses": [float(l) for l in losses],
                    }

                # Determine winner (among non-inf losses)
                valid_methods = {k: v["mean_loss"] for k, v in config_results.items()
                                 if v["mean_loss"] < 1e5}
                if valid_methods:
                    winner = min(valid_methods, key=valid_methods.get)
                    method_wins[winner] += 1
                    n_valid += 1
                    config_results["winner"] = winner
                else:
                    config_results["winner"] = "none"

                results[config_key] = config_results

                # Print progress
                losses_str = "  ".join(
                    f"{k}:{v['mean_loss']:.3f}" for k, v in config_results.items()
                    if k != "winner"
                )
                print(f"  {config_key}: {losses_str} -> {config_results['winner']}")

    # Summary
    print("\n" + "=" * 60)
    print(f"SUMMARY ({n_valid} valid configs)")
    print("=" * 60)
    for method, wins in sorted(method_wins.items(), key=lambda x: -x[1]):
        rate = wins / max(n_valid, 1) * 100
        print(f"  {method}: {wins}/{n_valid} ({rate:.1f}%)")

    results["summary"] = {
        "n_valid": n_valid,
        "wins": dict(method_wins),
        "win_rates": {k: v / max(n_valid, 1) for k, v in method_wins.items()},
    }

    os.makedirs("results/improved_baseline", exist_ok=True)
    with open("results/improved_baseline/baseline_results.json", "w") as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)

    print(f"\nResults saved to results/improved_baseline/baseline_results.json")
    return results


if __name__ == "__main__":
    run_improved_baseline()
