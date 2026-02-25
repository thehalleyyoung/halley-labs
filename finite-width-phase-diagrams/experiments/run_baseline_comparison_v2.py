"""
Expanded baseline comparison v2: PhaseKit vs LSUV vs Kaiming vs Xavier vs
Data-Dependent vs Gradient-Norm-Checking on 50+ configurations.

Compares all 6 initialization methods across:
- 5 depths × 3 widths × 4 activations = 60 configurations (hidden-layer sweep)
- Reports mean/std test loss, convergence rate, and head-to-head wins.
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from baselines import (
    xavier_init, kaiming_init, phasekit_init, lsuv_init,
    data_dependent_init, gradnorm_checking_init,
    gradient_norm_diagnostic, train_mlp, apply_activation,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj == float('inf') or obj == float('-inf'):
            return str(obj)
        return super().default(obj)


def generate_data(n_train=500, n_test=200, input_dim=10, seed=42):
    """Generate synthetic regression dataset with nonlinear target."""
    rng = np.random.RandomState(seed)
    X_train = rng.randn(n_train, input_dim)
    X_test = rng.randn(n_test, input_dim)
    def target(X):
        return (np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * X[:, 2]) +
                0.5 * X[:, 3] ** 2 - X[:, 4])
    y_train = target(X_train)
    y_test = target(X_test)
    mu, std = y_train.mean(), y_train.std() + 1e-8
    y_train = (y_train - mu) / std
    y_test = (y_test - mu) / std
    return X_train, y_train, X_test, y_test


def run_single_comparison(depth, width, activation, n_seeds=3, n_steps=500,
                          lr=0.01, X_train=None, y_train=None,
                          X_test=None, y_test=None):
    """Run one configuration across all 6 init methods and seeds."""
    input_dim = X_train.shape[1]
    dims = [input_dim] + [width] * (depth - 1) + [1]

    methods = {
        "xavier": lambda s: xavier_init(dims, seed=s),
        "kaiming": lambda s: kaiming_init(dims, activation, seed=s),
        "phasekit": lambda s: phasekit_init(dims, activation, seed=s),
        "lsuv": lambda s: lsuv_init(dims, activation, X_calib=X_train[:256], seed=s),
        "data_dep": lambda s: data_dependent_init(dims, activation, X_calib=X_train[:256], seed=s),
        "gradnorm": lambda s: gradnorm_checking_init(dims, activation, X_calib=X_train[:256], seed=s),
    }

    results = {}
    for method_name, init_fn in methods.items():
        seed_results = []
        for seed in range(n_seeds):
            try:
                init = init_fn(seed)
            except Exception as e:
                seed_results.append({
                    "final_loss": float('inf'), "test_loss": float('inf'),
                    "converged": False, "exploded": True, "error": str(e),
                })
                continue

            diag = gradient_norm_diagnostic(
                init.weights, init.biases, activation,
                X_train[:64], y_train[:64])

            train_result = train_mlp(
                init.weights, init.biases,
                X_train, y_train, X_test, y_test,
                activation=activation, n_steps=n_steps, lr=lr)

            train_result["vanishing"] = diag.vanishing
            train_result["exploding"] = diag.exploding
            train_result["sigma_w_per_layer"] = init.sigma_w_per_layer

            curve = train_result.pop("loss_curve", [])
            indices = list(range(0, len(curve), max(1, len(curve) // 20)))
            if len(curve) - 1 not in indices and len(curve) > 0:
                indices.append(len(curve) - 1)
            train_result["loss_curve_sampled"] = [curve[i] for i in indices]
            seed_results.append(train_result)

        final_losses = [r["final_loss"] for r in seed_results
                        if r.get("final_loss", float('inf')) != float('inf')]
        test_losses = [r["test_loss"] for r in seed_results
                       if r.get("test_loss", float('inf')) != float('inf')]
        converged = sum(1 for r in seed_results if r.get("converged", False))

        results[method_name] = {
            "mean_final_loss": float(np.mean(final_losses)) if final_losses else float('inf'),
            "std_final_loss": float(np.std(final_losses)) if final_losses else 0.0,
            "mean_test_loss": float(np.mean(test_losses)) if test_losses else float('inf'),
            "std_test_loss": float(np.std(test_losses)) if test_losses else 0.0,
            "converged_count": converged,
            "exploded_count": sum(1 for r in seed_results if r.get("exploded", False)),
            "n_seeds": n_seeds,
        }

    return results


def run_all_comparisons():
    """Run the full 60-config comparison suite."""
    print("=" * 70)
    print("EXPANDED BASELINE COMPARISON V2 (6 methods × 60 configs)")
    print("=" * 70)

    X_train, y_train, X_test, y_test = generate_data(n_train=500, n_test=200)
    all_results = {}

    depths = [3, 5, 8, 10, 15]
    widths = [32, 64, 128]
    activations = ["relu", "tanh", "gelu", "silu"]
    lr_map = {3: 0.005, 5: 0.003, 8: 0.001, 10: 0.001, 15: 0.0005}

    configs = []
    for depth in depths:
        for width in widths:
            for activation in activations:
                lr = lr_map[depth]
                label = f"{depth}L-{width}W-{activation}"
                configs.append((depth, width, activation, label, lr))

    print(f"Total configs: {len(configs)}")

    all_methods = ["xavier", "kaiming", "phasekit", "lsuv", "data_dep", "gradnorm"]
    win_counts = {m: 0 for m in all_methods}
    total_t0 = time.time()

    for idx, (depth, width, activation, label, lr) in enumerate(configs):
        print(f"\n[{idx+1}/{len(configs)}] {label} (lr={lr})...", end=" ", flush=True)
        t0 = time.time()
        results = run_single_comparison(
            depth, width, activation, n_seeds=3, n_steps=500,
            lr=lr, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test)
        elapsed = time.time() - t0

        losses = {m: results[m]["mean_test_loss"] for m in all_methods}
        best = min(losses, key=lambda k: losses[k])
        win_counts[best] += 1

        best_loss = losses[best]
        pk_loss = losses["phasekit"]
        print(f"best={best}({best_loss:.4f}) pk={pk_loss:.4f} [{elapsed:.1f}s]")

        all_results[label] = results

    total_elapsed = time.time() - total_t0

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Win counts (best mean test loss)")
    print("=" * 70)
    for m in all_methods:
        print(f"  {m:12s}: {win_counts[m]:3d} / {len(configs)}")

    print(f"\nTotal time: {total_elapsed:.0f}s")

    # Save
    output = {
        "n_configs": len(configs),
        "n_methods": len(all_methods),
        "methods": all_methods,
        "win_counts": win_counts,
        "configs": {label: {"depth": d, "width": w, "activation": a, "lr": lr}
                    for d, w, a, label, lr in configs},
        "results": all_results,
    }
    out_path = os.path.join(RESULTS_DIR, "baseline_comparison_v2.json")
    with open(out_path, 'w') as f:
        json.dump(output, f, cls=NumpyEncoder, indent=2)
    print(f"Results saved to {out_path}")

    return output


if __name__ == "__main__":
    run_all_comparisons()
