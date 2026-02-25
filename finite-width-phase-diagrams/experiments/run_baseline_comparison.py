"""
Head-to-head comparison: PhaseKit σ_w* vs LSUV vs Kaiming vs Xavier.

Compares initialization methods on:
1. 5-layer MLPs with different activations (ReLU, tanh, GELU, SiLU)
2. 10-layer deep MLPs (where initialization matters most)
3. Narrow networks (width 32-64) where finite-width corrections matter

Measures: training loss curve, final loss, test loss, gradient norms.
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from baselines import (
    xavier_init, kaiming_init, phasekit_init, lsuv_init,
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
    # Nonlinear target: sum of sinusoidal features + interaction terms
    def target(X):
        return (np.sin(X[:, 0] * 2) + np.cos(X[:, 1] * X[:, 2]) +
                0.5 * X[:, 3] ** 2 - X[:, 4])
    y_train = target(X_train)
    y_test = target(X_test)
    # Normalize
    mu, std = y_train.mean(), y_train.std() + 1e-8
    y_train = (y_train - mu) / std
    y_test = (y_test - mu) / std
    return X_train, y_train, X_test, y_test


def run_single_comparison(depth, width, activation, n_seeds=5, n_steps=500,
                          lr=0.01, X_train=None, y_train=None,
                          X_test=None, y_test=None):
    """Run one configuration across all init methods and seeds."""
    input_dim = X_train.shape[1]
    dims = [input_dim] + [width] * (depth - 1) + [1]

    methods = {
        "xavier": lambda s: xavier_init(dims, seed=s),
        "kaiming": lambda s: kaiming_init(dims, activation, seed=s),
        "phasekit": lambda s: phasekit_init(dims, activation, seed=s),
        "lsuv": lambda s: lsuv_init(dims, activation, X_calib=X_train[:256], seed=s),
    }

    results = {}
    for method_name, init_fn in methods.items():
        seed_results = []
        for seed in range(n_seeds):
            init = init_fn(seed)
            # Gradient diagnostic before training
            diag = gradient_norm_diagnostic(
                init.weights, init.biases, activation,
                X_train[:64], y_train[:64])

            # Train
            train_result = train_mlp(
                init.weights, init.biases,
                X_train, y_train, X_test, y_test,
                activation=activation, n_steps=n_steps, lr=lr)

            train_result["grad_norms_init"] = diag.layer_grad_norms
            train_result["act_norms_init"] = diag.layer_activation_norms
            train_result["vanishing"] = diag.vanishing
            train_result["exploding"] = diag.exploding
            train_result["sigma_w_per_layer"] = init.sigma_w_per_layer

            # Subsample loss curve for storage
            curve = train_result.pop("loss_curve", [])
            indices = list(range(0, len(curve), max(1, len(curve) // 50)))
            if len(curve) - 1 not in indices and len(curve) > 0:
                indices.append(len(curve) - 1)
            train_result["loss_curve_sampled"] = [curve[i] for i in indices]
            train_result["loss_curve_steps"] = indices
            seed_results.append(train_result)

        # Aggregate across seeds
        final_losses = [r["final_loss"] for r in seed_results
                        if r["final_loss"] != float('inf')]
        test_losses = [r["test_loss"] for r in seed_results
                       if r["test_loss"] != float('inf')]
        converged = sum(1 for r in seed_results if r["converged"])

        results[method_name] = {
            "mean_final_loss": float(np.mean(final_losses)) if final_losses else float('inf'),
            "std_final_loss": float(np.std(final_losses)) if final_losses else 0.0,
            "mean_test_loss": float(np.mean(test_losses)) if test_losses else float('inf'),
            "std_test_loss": float(np.std(test_losses)) if test_losses else 0.0,
            "converged_count": converged,
            "exploded_count": sum(1 for r in seed_results if r["exploded"]),
            "n_seeds": n_seeds,
            "per_seed": seed_results,
        }

    return results


def run_all_comparisons():
    """Run the full comparison suite."""
    print("=" * 60)
    print("HEAD-TO-HEAD BASELINE COMPARISON")
    print("=" * 60)

    X_train, y_train, X_test, y_test = generate_data(n_train=500, n_test=200)
    all_results = {}

    configs = [
        # (depth, width, activation, label, lr)
        # Shallow networks
        (5, 128, "relu",  "5L-128W-relu",  0.003),
        (5, 128, "tanh",  "5L-128W-tanh",  0.003),
        (5, 128, "gelu",  "5L-128W-gelu",  0.003),
        (5, 128, "silu",  "5L-128W-silu",  0.003),
        # Deep networks (where initialization matters most)
        (10, 128, "relu", "10L-128W-relu", 0.001),
        (10, 128, "gelu", "10L-128W-gelu", 0.001),
        (10, 128, "silu", "10L-128W-silu", 0.001),
        (10, 128, "tanh", "10L-128W-tanh", 0.001),
        # Narrow networks (finite-width corrections matter)
        (5, 32, "gelu",   "5L-32W-gelu",  0.003),
        (5, 64, "gelu",   "5L-64W-gelu",  0.003),
        (10, 32, "gelu", "10L-32W-gelu",  0.001),
        (10, 64, "gelu", "10L-64W-gelu",  0.001),
        # Deep narrow (hardest for initialization)
        (15, 64, "gelu", "15L-64W-gelu",  0.0005),
        (15, 64, "silu", "15L-64W-silu",  0.0005),
    ]

    for depth, width, activation, label, lr in configs:
        print(f"\n--- Config: {label} (depth={depth}, width={width}, act={activation}, lr={lr}) ---")
        t0 = time.time()
        results = run_single_comparison(
            depth, width, activation, n_seeds=5, n_steps=500,
            lr=lr, X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test)
        elapsed = time.time() - t0

        for method, r in results.items():
            status = f"test_loss={r['mean_test_loss']:.4f}±{r['std_test_loss']:.4f}"
            conv = f"converged={r['converged_count']}/{r['n_seeds']}"
            expl = f"exploded={r['exploded_count']}"
            print(f"  {method:10s}: {status}, {conv}, {expl}")

        all_results[label] = results
        print(f"  ({elapsed:.1f}s)")

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: Mean test loss by method (lower is better)")
    print("=" * 60)
    print(f"{'Config':<20s} {'Xavier':>10s} {'Kaiming':>10s} {'PhaseKit':>10s} {'LSUV':>10s} {'Best':>10s}")
    print("-" * 70)

    win_counts = {"xavier": 0, "kaiming": 0, "phasekit": 0, "lsuv": 0}
    for label, results in all_results.items():
        losses = {}
        for m in ["xavier", "kaiming", "phasekit", "lsuv"]:
            losses[m] = results[m]["mean_test_loss"]

        best = min(losses, key=lambda k: losses[k])
        win_counts[best] += 1

        cols = [f"{losses[m]:.4f}" if losses[m] < float('inf') else "  inf" for m in
                ["xavier", "kaiming", "phasekit", "lsuv"]]
        print(f"{label:<20s} {cols[0]:>10s} {cols[1]:>10s} {cols[2]:>10s} {cols[3]:>10s} {best:>10s}")

    print("-" * 70)
    print(f"{'Win count':<20s}", end="")
    for m in ["xavier", "kaiming", "phasekit", "lsuv"]:
        print(f"{win_counts[m]:>10d}", end="")
    print()

    # Compute PhaseKit advantage ratios
    advantage = {}
    for label, results in all_results.items():
        pk = results["phasekit"]["mean_test_loss"]
        if pk > 0 and pk < float('inf'):
            advantage[label] = {
                "vs_xavier": results["xavier"]["mean_test_loss"] / pk,
                "vs_kaiming": results["kaiming"]["mean_test_loss"] / pk,
                "vs_lsuv": results["lsuv"]["mean_test_loss"] / pk,
            }

    all_results["_summary"] = {
        "win_counts": win_counts,
        "phasekit_advantage_ratios": advantage,
        "total_configs": len(configs),
    }

    # Save
    output_path = os.path.join(RESULTS_DIR, 'exp_baseline_comparison.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")

    return all_results


if __name__ == "__main__":
    run_all_comparisons()
