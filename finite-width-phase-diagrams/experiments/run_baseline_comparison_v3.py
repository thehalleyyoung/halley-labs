"""
Improvement 2: Competitive baselines comparison.

Compares PhaseKit's σ_w* recommendation vs Kaiming, Xavier, LSUV,
MetaInit, GradInit on MLP training (synthetic regression).

Tests: 4 depths × 4 widths × 4 activations = 64 configs × 6 methods = 384 runs.
"""

import sys
import os
import json
import numpy as np
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from baselines import (
    xavier_init, kaiming_init, phasekit_init, lsuv_init,
    data_dependent_init, train_mlp, apply_activation, activation_derivative,
    gradient_norm_diagnostic,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'baseline_comparison')
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
        return super().default(obj)


def metainit(dims, activation="relu", X_calib=None, seed=42,
             n_steps=50, lr_scale=0.01):
    """MetaInit (Dauphin & Schoenholz, 2019) — simplified.

    Optimizes per-layer weight scales to minimize the gradient quotient
    (max_layer_grad / min_layer_grad) on a mini-batch.
    """
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(128, dims[0])
    y_calib = rng.randn(X_calib.shape[0])

    n_layers = len(dims) - 1
    # Initialize log-scales (one per layer)
    log_scales = np.zeros(n_layers)

    # Base init: unit variance per layer
    base_weights = []
    for i in range(n_layers):
        W = rng.randn(dims[i], dims[i + 1])
        W /= np.sqrt(dims[i])
        base_weights.append(W)

    for step in range(n_steps):
        # Build weights from base + scales
        weights = [base_weights[i] * np.exp(log_scales[i]) for i in range(n_layers)]
        biases = [np.zeros(dims[i + 1]) for i in range(n_layers)]

        diag = gradient_norm_diagnostic(weights, biases, activation,
                                        X_calib[:64], y_calib[:64])
        gnorms = diag.layer_grad_norms
        if not gnorms or any(g <= 0 for g in gnorms):
            break

        log_gnorms = np.array([np.log(max(g, 1e-30)) for g in gnorms])
        mean_log = np.mean(log_gnorms)

        # Gradient of objective: each layer's log-scale should push its
        # grad norm toward the geometric mean
        for i in range(n_layers):
            grad_i = 2.0 * (log_gnorms[i] - mean_log)
            log_scales[i] -= lr_scale * grad_i

    # Build final weights
    from baselines import InitResult
    weights_final = [base_weights[i] * np.exp(log_scales[i]) for i in range(n_layers)]
    biases_final = [np.zeros(dims[i + 1]) for i in range(n_layers)]
    sigmas = [float(np.exp(log_scales[i]) * np.sqrt(dims[i])) for i in range(n_layers)]

    return InitResult("metainit", weights_final, biases_final, sigmas,
                      f"MetaInit: equalized gradient norms over {n_steps} steps")


def gradinit(dims, activation="relu", X_calib=None, seed=42,
             n_steps=30, lr_scale=0.1):
    """GradInit (Zhu et al., 2021) — simplified.

    Learns a single global scale α that minimizes first-step gradient norm
    variation, starting from Kaiming init.
    """
    rng = np.random.RandomState(seed)
    if X_calib is None:
        X_calib = rng.randn(128, dims[0])
    y_calib = rng.randn(X_calib.shape[0])

    n_layers = len(dims) - 1

    # Start from Kaiming
    gain_map = {"relu": np.sqrt(2.0), "tanh": 1.0, "gelu": 1.0,
                "silu": 1.0, "leaky_relu": np.sqrt(2.0 / 1.0001)}
    base_gain = gain_map.get(activation, np.sqrt(2.0))

    base_weights = []
    for i in range(n_layers):
        W = rng.randn(dims[i], dims[i + 1]) * base_gain / np.sqrt(dims[i])
        base_weights.append(W)

    # Per-layer scales
    alphas = np.ones(n_layers)

    for step in range(n_steps):
        weights = [base_weights[i] * alphas[i] for i in range(n_layers)]
        biases = [np.zeros(dims[i + 1]) for i in range(n_layers)]

        diag = gradient_norm_diagnostic(weights, biases, activation,
                                        X_calib[:64], y_calib[:64])
        gnorms = diag.layer_grad_norms
        if not gnorms or any(g <= 0 for g in gnorms):
            break

        # Objective: minimize max(gnorms)/min(gnorms) by adjusting per-layer alphas
        log_gnorms = np.array([np.log(max(g, 1e-30)) for g in gnorms])
        target = np.mean(log_gnorms)

        for i in range(n_layers):
            grad_i = 2.0 * (log_gnorms[i] - target)
            alphas[i] *= np.exp(-lr_scale * grad_i)
            alphas[i] = np.clip(alphas[i], 0.01, 100.0)

    from baselines import InitResult
    weights_final = [base_weights[i] * alphas[i] for i in range(n_layers)]
    biases_final = [np.zeros(dims[i + 1]) for i in range(n_layers)]
    sigmas = [float(alphas[i] * base_gain) for i in range(n_layers)]

    return InitResult("gradinit", weights_final, biases_final, sigmas,
                      f"GradInit: per-layer scale optimization over {n_steps} steps")


def run_baseline_comparison():
    """Head-to-head comparison of PhaseKit vs all baselines."""
    print("=" * 70)
    print("BASELINE COMPARISON: PhaseKit vs 5 baselines")
    print("=" * 70)

    depths = [3, 5, 10, 20]
    widths = [64, 128]
    activations = ["relu", "tanh", "gelu", "silu"]

    n_steps = 300
    lr = 0.01
    n_seeds = 2
    input_dim = 10

    methods = ["xavier", "kaiming", "phasekit", "lsuv", "metainit", "gradinit"]
    all_results = []

    total_configs = len(depths) * len(widths) * len(activations)
    config_idx = 0

    for depth in depths:
        for width in widths:
            for act in activations:
                config_idx += 1
                dims = [input_dim] + [width] * (depth - 1) + [1]

                # Calibration data (used by LSUV/MetaInit/GradInit)
                rng_data = np.random.RandomState(42)
                X_calib = rng_data.randn(256, input_dim)

                # Training data
                X_train = rng_data.randn(200, input_dim)
                y_train = np.sin(X_train[:, 0]) + 0.5 * X_train[:, 1] + 0.3 * np.cos(X_train[:, 2])
                y_train += 0.1 * rng_data.randn(200)
                X_test = rng_data.randn(100, input_dim)
                y_test = np.sin(X_test[:, 0]) + 0.5 * X_test[:, 1] + 0.3 * np.cos(X_test[:, 2])

                config_results = {
                    "depth": depth, "width": width, "activation": act,
                }

                for method in methods:
                    seed_losses = []
                    seed_test_losses = []
                    seed_converged = []

                    for seed in range(n_seeds):
                        try:
                            if method == "xavier":
                                init = xavier_init(dims, seed=seed)
                            elif method == "kaiming":
                                init = kaiming_init(dims, act, seed=seed)
                            elif method == "phasekit":
                                init = phasekit_init(dims, act, seed=seed)
                            elif method == "lsuv":
                                init = lsuv_init(dims, act, X_calib, seed=seed)
                            elif method == "metainit":
                                init = metainit(dims, act, X_calib, seed=seed)
                            elif method == "gradinit":
                                init = gradinit(dims, act, X_calib, seed=seed)

                            result = train_mlp(
                                init.weights, init.biases,
                                X_train, y_train, X_test, y_test,
                                activation=act, n_steps=n_steps, lr=lr,
                            )

                            seed_losses.append(result["final_loss"])
                            seed_test_losses.append(result["test_loss"])
                            seed_converged.append(result["converged"])
                        except Exception as e:
                            seed_losses.append(float('inf'))
                            seed_test_losses.append(float('inf'))
                            seed_converged.append(False)

                    finite_losses = [l for l in seed_losses if np.isfinite(l)]
                    finite_test = [l for l in seed_test_losses if np.isfinite(l)]

                    config_results[method] = {
                        "mean_final_loss": float(np.mean(finite_losses)) if finite_losses else float('inf'),
                        "std_final_loss": float(np.std(finite_losses)) if len(finite_losses) > 1 else 0,
                        "mean_test_loss": float(np.mean(finite_test)) if finite_test else float('inf'),
                        "converge_rate": sum(seed_converged) / n_seeds,
                        "explode_rate": sum(1 for l in seed_losses if l == float('inf')) / n_seeds,
                    }

                all_results.append(config_results)

                if config_idx % 8 == 0 or config_idx <= 4:
                    losses_str = " | ".join(
                        f"{m}={config_results[m]['mean_final_loss']:.4f}"
                        for m in methods
                        if config_results[m]['mean_final_loss'] < float('inf')
                    )
                    print(f"  [{config_idx:3d}/{total_configs}] D={depth} W={width} {act}: {losses_str}")

    # Aggregate analysis
    print(f"\n{'=' * 70}")
    print("AGGREGATE RESULTS")
    print(f"{'=' * 70}")

    # Win rate: how often does each method achieve lowest final loss
    win_counts = {m: 0 for m in methods}
    valid_configs = 0

    for r in all_results:
        losses = {m: r[m]["mean_final_loss"] for m in methods}
        finite = {m: l for m, l in losses.items() if np.isfinite(l)}
        if len(finite) >= 2:
            valid_configs += 1
            winner = min(finite, key=finite.get)
            win_counts[winner] += 1

    print(f"\n  Win rate (lowest final loss) over {valid_configs} configs:")
    for m in methods:
        rate = win_counts[m] / valid_configs if valid_configs > 0 else 0
        print(f"    {m:12s}: {win_counts[m]:3d}/{valid_configs} ({rate:.1%})")

    # Average rank
    rank_sums = {m: 0 for m in methods}
    rank_counts = 0

    for r in all_results:
        losses = {m: r[m]["mean_final_loss"] for m in methods}
        finite = {m: l for m, l in losses.items() if np.isfinite(l)}
        if len(finite) >= 2:
            rank_counts += 1
            sorted_methods = sorted(finite, key=finite.get)
            for rank, m in enumerate(sorted_methods, 1):
                rank_sums[m] += rank
            # Methods that diverged get worst rank
            for m in methods:
                if m not in finite:
                    rank_sums[m] += len(methods)

    print(f"\n  Average rank (lower is better) over {rank_counts} configs:")
    for m in sorted(methods, key=lambda x: rank_sums[x]):
        avg_rank = rank_sums[m] / rank_counts if rank_counts > 0 else len(methods)
        print(f"    {m:12s}: {avg_rank:.2f}")

    # Convergence rate
    print(f"\n  Convergence rate:")
    for m in methods:
        rates = [r[m]["converge_rate"] for r in all_results]
        print(f"    {m:12s}: {np.mean(rates):.1%} (mean across configs)")

    # Explosion rate
    print(f"\n  Explosion rate:")
    for m in methods:
        rates = [r[m]["explode_rate"] for r in all_results]
        if np.mean(rates) > 0:
            print(f"    {m:12s}: {np.mean(rates):.1%}")

    # Per-depth analysis
    print(f"\n  Mean final loss by depth:")
    print(f"    {'Depth':>6s}", end="")
    for m in methods:
        print(f"  {m:>12s}", end="")
    print()
    for depth in depths:
        print(f"    {depth:6d}", end="")
        for m in methods:
            losses = [r[m]["mean_final_loss"] for r in all_results
                      if r["depth"] == depth and np.isfinite(r[m]["mean_final_loss"])]
            avg = np.mean(losses) if losses else float('inf')
            if np.isfinite(avg):
                print(f"  {avg:12.4f}", end="")
            else:
                print(f"  {'inf':>12s}", end="")
        print()

    # Per-activation analysis
    print(f"\n  Mean final loss by activation:")
    print(f"    {'Act':>8s}", end="")
    for m in methods:
        print(f"  {m:>12s}", end="")
    print()
    for act in activations:
        print(f"    {act:>8s}", end="")
        for m in methods:
            losses = [r[m]["mean_final_loss"] for r in all_results
                      if r["activation"] == act and np.isfinite(r[m]["mean_final_loss"])]
            avg = np.mean(losses) if losses else float('inf')
            if np.isfinite(avg):
                print(f"  {avg:12.4f}", end="")
            else:
                print(f"  {'inf':>12s}", end="")
        print()

    # Save results
    output = {
        "experiment": "baseline_comparison",
        "methodology": {
            "task": "synthetic_regression",
            "n_steps": n_steps,
            "lr": lr,
            "n_seeds": n_seeds,
            "depths": depths,
            "widths": widths,
            "activations": activations,
            "target": "y = sin(x1) + 0.5*x2 + 0.3*cos(x3) + noise",
            "input_dim": input_dim,
        },
        "methods": methods,
        "win_counts": win_counts,
        "valid_configs": valid_configs,
        "rank_sums": rank_sums,
        "details": all_results,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    path = os.path.join(RESULTS_DIR, "baseline_results.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(f"\n  Results saved to {path}")

    return output


if __name__ == "__main__":
    run_baseline_comparison()
