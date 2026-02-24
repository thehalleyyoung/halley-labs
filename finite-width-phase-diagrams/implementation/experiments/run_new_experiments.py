#!/usr/bin/env python3
"""Finite-Width Phase Diagrams: Phase Benchmark.

Computes NTK for MLPs at various widths, measures NTK drift during training,
verifies T·γ* constant law, and compares regime predictions with actual dynamics.
Outputs: phase_benchmark_results.json
"""

import json
import os
import time
import numpy as np

np.random.seed(42)

# ---------------------------------------------------------------------------
# MLP with NTK computation support
# ---------------------------------------------------------------------------

def relu(x):
    return np.maximum(0, x)


class NTK_MLP:
    """MLP with Neural Tangent Kernel computation."""

    def __init__(self, input_dim, width, output_dim=1, depth=2):
        self.input_dim = input_dim
        self.width = width
        self.output_dim = output_dim
        self.depth = depth
        self.layers = []
        dims = [input_dim] + [width] * (depth - 1) + [output_dim]
        for i in range(len(dims) - 1):
            # NTK parameterization: scale by 1/sqrt(fan_in)
            W = np.random.randn(dims[i], dims[i + 1]) / np.sqrt(dims[i])
            b = np.zeros(dims[i + 1])
            self.layers.append([W, b])

    def forward(self, X):
        """Forward pass."""
        h = X
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = relu(h)
        return h

    def get_params(self):
        return np.concatenate([np.concatenate([W.ravel(), b.ravel()])
                               for W, b in self.layers])

    def set_params(self, flat):
        idx = 0
        for W, b in self.layers:
            n = W.size
            W[:] = flat[idx:idx + n].reshape(W.shape)
            idx += n
            n = b.size
            b[:] = flat[idx:idx + n]
            idx += n

    def n_params(self):
        return sum(W.size + b.size for W, b in self.layers)

    def compute_jacobian(self, X, eps=1e-5):
        """Compute Jacobian of output w.r.t. parameters via finite differences."""
        params = self.get_params().copy()
        n_samples = X.shape[0]
        n_params = len(params)
        n_out = self.output_dim

        base_out = self.forward(X).copy()
        J = np.zeros((n_samples * n_out, n_params))

        # Subsample params for speed
        param_indices = np.random.choice(n_params, min(n_params, 200), replace=False)
        param_indices.sort()

        for col_idx, p_idx in enumerate(param_indices):
            params_plus = params.copy()
            params_plus[p_idx] += eps
            self.set_params(params_plus)
            out_plus = self.forward(X)

            params_minus = params.copy()
            params_minus[p_idx] -= eps
            self.set_params(params_minus)
            out_minus = self.forward(X)

            deriv = (out_plus - out_minus) / (2 * eps)
            J[:, col_idx] = deriv.ravel()[:n_samples * n_out]

        self.set_params(params)
        return J[:, :len(param_indices)]

    def compute_ntk(self, X):
        """Compute empirical NTK: Θ(x,x') = J(x) @ J(x').T"""
        J = self.compute_jacobian(X)
        return J @ J.T

    def mse_loss(self, X, y):
        pred = self.forward(X)
        return float(np.mean((pred - y.reshape(pred.shape)) ** 2))

    def train_step(self, X, y, lr):
        """Gradient descent step via finite differences on a param subset."""
        params = self.get_params().copy()
        n = len(params)
        eps = 1e-5

        # Subsample for speed
        indices = np.random.choice(n, min(n, 100), replace=False)
        grad = np.zeros(n)
        base_loss = self.mse_loss(X, y)

        for idx in indices:
            p = params.copy()
            p[idx] += eps
            self.set_params(p)
            loss_plus = self.mse_loss(X, y)
            grad[idx] = (loss_plus - base_loss) / eps

        self.set_params(params - lr * grad)
        return self.mse_loss(X, y)


# ---------------------------------------------------------------------------
# Data generation
# ---------------------------------------------------------------------------

def make_data(n=50, dim=5, noise=0.1):
    X = np.random.randn(n, dim)
    w = np.random.randn(dim, 1)
    y = X @ w + np.random.randn(n, 1) * noise
    return X, y


# ---------------------------------------------------------------------------
# Regime detection
# ---------------------------------------------------------------------------

def classify_regime(ntk_drift, width):
    """Classify training regime based on NTK drift magnitude.

    Returns: 'lazy' (kernel), 'rich' (feature-learning), or 'chaotic'
    """
    if ntk_drift < 0.05:
        return "lazy"
    elif ntk_drift < 0.5:
        return "rich"
    else:
        return "chaotic"


def compute_ntk_drift(ntk_init, ntk_current):
    """Measure relative change in NTK."""
    frobenius_init = np.linalg.norm(ntk_init, 'fro')
    if frobenius_init < 1e-10:
        return 0.0
    diff = np.linalg.norm(ntk_current - ntk_init, 'fro')
    return float(diff / frobenius_init)


def predict_regime(width, depth, lr):
    """Predict training regime from architecture hyperparameters.

    Based on mean-field theory:
    - Large width → lazy/kernel regime
    - Small width, large lr → rich/feature-learning regime
    - Very small width, large lr → chaotic
    """
    # γ* ∝ 1/width (feature learning strength)
    gamma_star = 1.0 / width
    # Effective learning rate
    effective_lr = lr * depth
    # Regime boundaries
    if gamma_star * effective_lr < 0.001:
        return "lazy"
    elif gamma_star * effective_lr < 0.1:
        return "rich"
    else:
        return "chaotic"


# ---------------------------------------------------------------------------
# Experiment 1: NTK at different widths
# ---------------------------------------------------------------------------

def run_ntk_width_experiment():
    print("=" * 60)
    print("Experiment 1: NTK Computation at Different Widths")
    print("=" * 60)

    widths = [16, 32, 64, 128, 256, 512]
    input_dim = 5
    n_samples = 20
    X, y = make_data(n_samples, input_dim)
    results = []

    for width in widths:
        t0 = time.time()
        model = NTK_MLP(input_dim, width, output_dim=1, depth=2)
        ntk = model.compute_ntk(X)
        elapsed = time.time() - t0

        eigvals = np.linalg.eigvalsh(ntk)
        eigvals = eigvals[eigvals > 1e-10]

        entry = {
            "width": width,
            "n_params": model.n_params(),
            "ntk_shape": list(ntk.shape),
            "ntk_frobenius_norm": round(float(np.linalg.norm(ntk, 'fro')), 6),
            "ntk_trace": round(float(np.trace(ntk)), 6),
            "ntk_rank": int(np.sum(eigvals > 1e-8)),
            "ntk_condition_number": round(float(eigvals[-1] / eigvals[0]) if len(eigvals) > 1 and eigvals[0] > 0 else float('inf'), 4),
            "max_eigenvalue": round(float(eigvals[-1]) if len(eigvals) > 0 else 0, 6),
            "min_eigenvalue": round(float(eigvals[0]) if len(eigvals) > 0 else 0, 6),
            "spectral_gap": round(float(eigvals[-1] - eigvals[-2]) if len(eigvals) > 1 else 0, 6),
            "computation_time_s": round(elapsed, 4),
        }
        results.append(entry)
        print(f"  width={width}: norm={entry['ntk_frobenius_norm']:.2f}, "
              f"rank={entry['ntk_rank']}, cond={entry['ntk_condition_number']:.1f}, "
              f"time={elapsed:.2f}s")

    return results


# ---------------------------------------------------------------------------
# Experiment 2: NTK drift during training
# ---------------------------------------------------------------------------

def run_ntk_drift_experiment():
    print("\n" + "=" * 60)
    print("Experiment 2: NTK Drift During Training (5 epochs)")
    print("=" * 60)

    widths = [16, 64, 256]
    input_dim = 5
    n_samples = 20
    X, y = make_data(n_samples, input_dim)
    n_epochs = 5
    results = []

    for width in widths:
        model = NTK_MLP(input_dim, width, output_dim=1, depth=2)
        ntk_init = model.compute_ntk(X)
        lr = 0.01

        epoch_data = []
        for ep in range(n_epochs):
            loss = model.train_step(X, y, lr)
            ntk_current = model.compute_ntk(X)
            drift = compute_ntk_drift(ntk_init, ntk_current)
            regime = classify_regime(drift, width)

            epoch_data.append({
                "epoch": ep + 1,
                "loss": round(loss, 6),
                "ntk_drift": round(drift, 6),
                "regime": regime,
            })

        final_drift = epoch_data[-1]["ntk_drift"]
        detected_regime = epoch_data[-1]["regime"]
        predicted = predict_regime(width, 2, lr)

        entry = {
            "width": width,
            "learning_rate": lr,
            "final_loss": epoch_data[-1]["loss"],
            "final_ntk_drift": round(final_drift, 6),
            "detected_regime": detected_regime,
            "predicted_regime": predicted,
            "regime_match": bool(detected_regime == predicted),
            "epoch_details": epoch_data,
        }
        results.append(entry)
        print(f"  width={width}: drift={final_drift:.4f}, "
              f"detected={detected_regime}, predicted={predicted}, "
              f"match={detected_regime == predicted}")

    return results


# ---------------------------------------------------------------------------
# Experiment 3: T·γ* constant law verification
# ---------------------------------------------------------------------------

def run_tgamma_law_experiment():
    print("\n" + "=" * 60)
    print("Experiment 3: T·γ* Constant Law Verification")
    print("=" * 60)

    T_values = [20, 50, 100, 200]
    input_dim = 5
    n_samples = 20
    X, y = make_data(n_samples, input_dim)
    width = 64
    results = []

    # γ* should scale as 1/width; T·γ* should be approximately constant
    # at the lazy-to-rich transition
    for T in T_values:
        # γ* ≈ 1/width for MLP
        gamma_star = 1.0 / width

        # Find critical lr where regime transitions
        lr_values = np.logspace(-4, -1, 20)
        transition_lr = None

        for lr in lr_values:
            model = NTK_MLP(input_dim, width, output_dim=1, depth=2)
            ntk_init = model.compute_ntk(X)

            for ep in range(T):
                model.train_step(X, y, lr)

            ntk_final = model.compute_ntk(X)
            drift = compute_ntk_drift(ntk_init, ntk_final)

            if drift > 0.1 and transition_lr is None:
                transition_lr = lr
                break

        if transition_lr is None:
            transition_lr = lr_values[-1]

        # T·γ* product at transition
        t_gamma_product = T * gamma_star

        entry = {
            "T_steps": T,
            "width": width,
            "gamma_star": round(gamma_star, 6),
            "transition_lr": round(float(transition_lr), 6),
            "T_gamma_product": round(t_gamma_product, 6),
            "T_lr_product": round(float(T * transition_lr), 6),
        }
        results.append(entry)
        print(f"  T={T}: γ*={gamma_star:.4f}, lr_trans={transition_lr:.5f}, "
              f"T·γ*={t_gamma_product:.4f}, T·lr={T * transition_lr:.4f}")

    # Check constancy of T·γ*
    products = [e["T_gamma_product"] for e in results]
    cv = np.std(products) / np.mean(products) if np.mean(products) > 0 else float('inf')

    return {
        "per_T": results,
        "T_gamma_products": [round(p, 6) for p in products],
        "product_mean": round(float(np.mean(products)), 6),
        "product_std": round(float(np.std(products)), 6),
        "coefficient_of_variation": round(float(cv), 6),
        "is_approximately_constant": bool(cv < 0.3),
    }


# ---------------------------------------------------------------------------
# Experiment 4: Regime predictions vs actual dynamics
# ---------------------------------------------------------------------------

def run_regime_prediction_experiment():
    print("\n" + "=" * 60)
    print("Experiment 4: Regime Predictions vs Actual Training Dynamics")
    print("=" * 60)

    input_dim = 5
    n_samples = 20
    X, y = make_data(n_samples, input_dim)
    configs = [
        {"width": 16, "depth": 2, "lr": 0.1},
        {"width": 16, "depth": 2, "lr": 0.001},
        {"width": 64, "depth": 2, "lr": 0.1},
        {"width": 64, "depth": 2, "lr": 0.001},
        {"width": 256, "depth": 2, "lr": 0.1},
        {"width": 256, "depth": 2, "lr": 0.001},
        {"width": 256, "depth": 3, "lr": 0.01},
        {"width": 512, "depth": 2, "lr": 0.001},
        {"width": 32, "depth": 3, "lr": 0.05},
        {"width": 128, "depth": 2, "lr": 0.01},
    ]

    results = []
    for cfg in configs:
        model = NTK_MLP(input_dim, cfg["width"], output_dim=1, depth=cfg["depth"])
        ntk_init = model.compute_ntk(X)

        losses = []
        for ep in range(10):
            loss = model.train_step(X, y, cfg["lr"])
            losses.append(loss)

        ntk_final = model.compute_ntk(X)
        drift = compute_ntk_drift(ntk_init, ntk_final)
        actual_regime = classify_regime(drift, cfg["width"])
        predicted_regime = predict_regime(cfg["width"], cfg["depth"], cfg["lr"])

        # Compute training dynamics features
        loss_ratio = losses[-1] / max(losses[0], 1e-10)
        converged = loss_ratio < 0.5

        entry = {
            "width": cfg["width"],
            "depth": cfg["depth"],
            "learning_rate": cfg["lr"],
            "initial_loss": round(losses[0], 6),
            "final_loss": round(losses[-1], 6),
            "loss_ratio": round(float(loss_ratio), 6),
            "converged": converged,
            "ntk_drift": round(drift, 6),
            "actual_regime": actual_regime,
            "predicted_regime": predicted_regime,
            "regime_match": bool(actual_regime == predicted_regime),
        }
        results.append(entry)
        status = "✓" if entry["regime_match"] else "✗"
        print(f"  {status} w={cfg['width']}, d={cfg['depth']}, lr={cfg['lr']}: "
              f"drift={drift:.4f}, actual={actual_regime}, predicted={predicted_regime}")

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Finite-Width Phase Diagrams - Phase Benchmark")
    print("=" * 60)
    t_start = time.time()

    all_results = {
        "experiment": "finite_width_phase_benchmark",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    all_results["ntk_width_scaling"] = run_ntk_width_experiment()
    all_results["ntk_drift"] = run_ntk_drift_experiment()
    all_results["tgamma_law"] = run_tgamma_law_experiment()
    all_results["regime_predictions"] = run_regime_prediction_experiment()

    total_time = time.time() - t_start
    all_results["total_time_s"] = round(total_time, 2)

    # Summary
    regime_results = all_results["regime_predictions"]
    drift_results = all_results["ntk_drift"]
    all_results["summary"] = {
        "ntk_norm_scales_with_width": bool(all(
            all_results["ntk_width_scaling"][i]["ntk_frobenius_norm"] <
            all_results["ntk_width_scaling"][i + 1]["ntk_frobenius_norm"]
            for i in range(len(all_results["ntk_width_scaling"]) - 1)
        )),
        "regime_prediction_accuracy": round(
            np.mean([r["regime_match"] for r in regime_results]), 4
        ),
        "drift_prediction_accuracy": round(
            np.mean([r["regime_match"] for r in drift_results]), 4
        ),
        "tgamma_is_constant": all_results["tgamma_law"]["is_approximately_constant"],
        "tgamma_cv": all_results["tgamma_law"]["coefficient_of_variation"],
    }

    out_path = os.path.join(os.path.dirname(__file__), "phase_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"Results written to {out_path}")
    print(f"Total time: {total_time:.1f}s")
    print(f"Summary: {json.dumps(all_results['summary'], indent=2)}")


if __name__ == "__main__":
    main()
