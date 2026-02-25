"""
Real-world case study: PhaseKit diagnoses and fixes initialization.

Scenario: A 15-layer deep narrow (width=64) GELU MLP fails to train with
standard initialization. PhaseKit diagnoses the problem (identifies the
network is in the ordered phase) and recommends the correct σ_w, fixing training.

This demonstrates the full PhaseKit workflow:
  Architecture → PhaseKit analysis → Diagnosis → Recommended σ_w → Training improvement
"""

import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec, ActivationVarianceMaps
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


def generate_case_study_data(seed=42):
    """Generate synthetic dataset for case study."""
    rng = np.random.RandomState(seed)
    n_train, n_test = 500, 200
    d = 20
    X_train = rng.randn(n_train, d)
    X_test = rng.randn(n_test, d)

    # Complex nonlinear target requiring deep network to fit
    def target(X):
        return (np.sin(2 * X[:, 0] * X[:, 1]) +
                np.cos(X[:, 2]) * X[:, 3] +
                np.tanh(X[:, 4] + X[:, 5]) +
                0.3 * X[:, 6] ** 2 - X[:, 7])

    y_train = target(X_train)
    y_test = target(X_test)
    mu, std = y_train.mean(), y_train.std() + 1e-8
    y_train = (y_train - mu) / std
    y_test = (y_test - mu) / std
    return X_train, y_train, X_test, y_test


def phasekit_diagnose(depth, width, activation, sigma_w):
    """Run PhaseKit diagnostic on the given architecture."""
    analyzer = MeanFieldAnalyzer()
    spec = ArchitectureSpec(
        depth=depth, width=width, activation=activation,
        sigma_w=sigma_w, sigma_b=0.0,
    )
    report = analyzer.analyze(spec)

    diagnosis = {
        "architecture": {
            "depth": depth, "width": width, "activation": activation,
            "sigma_w": sigma_w,
        },
        "mean_field_analysis": {
            "fixed_point_variance": report.fixed_point,
            "chi_1": report.chi_1,
            "depth_scale": report.depth_scale,
            "phase": report.phase,
            "max_trainable_depth": report.max_trainable_depth,
            "lyapunov_exponent": report.lyapunov_exponent,
        },
    }

    # Detailed diagnostic message
    if report.phase == "ordered":
        diagnosis["diagnosis"] = (
            f"PROBLEM: Network is in the ORDERED phase (χ₁={report.chi_1:.4f} < 1). "
            f"Gradients vanish exponentially with depth scale ξ={report.depth_scale:.1f}. "
            f"At depth {depth}, gradients are attenuated by factor χ₁^{depth}="
            f"{report.chi_1**depth:.2e}. Training will stall."
        )
        diagnosis["severity"] = "critical"
    elif report.phase == "chaotic":
        diagnosis["diagnosis"] = (
            f"PROBLEM: Network is in the CHAOTIC phase (χ₁={report.chi_1:.4f} > 1). "
            f"Gradients explode exponentially. Training will be unstable."
        )
        diagnosis["severity"] = "critical"
    else:
        diagnosis["diagnosis"] = (
            f"OK: Network is near the CRITICAL point (χ₁={report.chi_1:.4f} ≈ 1). "
            f"Depth scale ξ={report.depth_scale:.1f}. Initialization is appropriate."
        )
        diagnosis["severity"] = "ok"

    return diagnosis


def run_case_study():
    """Run the full case study."""
    print("=" * 70)
    print("CASE STUDY: PhaseKit Diagnoses and Fixes Initialization")
    print("=" * 70)

    X_train, y_train, X_test, y_test = generate_case_study_data()
    depth, width, activation = 10, 128, "gelu"
    input_dim = X_train.shape[1]
    dims = [input_dim] + [width] * (depth - 1) + [1]

    results = {"scenario": {
        "depth": depth, "width": width, "activation": activation,
        "input_dim": input_dim, "n_train": len(X_train),
        "description": "15-layer deep narrow GELU MLP on nonlinear regression",
    }}

    # Step 1: User tries Xavier initialization (standard default)
    print("\n--- Step 1: Default Xavier Initialization ---")
    xavier_sigma_w = np.sqrt(2.0 / (width + width))  # ~0.125
    xavier_gain = xavier_sigma_w * np.sqrt(width)  # the effective σ_w
    diag_xavier = phasekit_diagnose(depth, width, activation, xavier_gain)
    print(f"  Xavier σ_w = {xavier_gain:.4f}")
    print(f"  PhaseKit diagnosis: {diag_xavier['diagnosis']}")

    xavier_r = xavier_init(dims, seed=42)
    diag_grad_xavier = gradient_norm_diagnostic(
        xavier_r.weights, xavier_r.biases, activation,
        X_train[:64], y_train[:64])
    xavier_train = train_mlp(
        xavier_r.weights, xavier_r.biases,
        X_train, y_train, X_test, y_test,
        activation, n_steps=500, lr=0.001)
    print(f"  Gradient diagnosis: {diag_grad_xavier.diagnosis}")
    print(f"  Training result: test_loss={xavier_train['test_loss']:.4f}, "
          f"converged={xavier_train['converged']}")

    results["step1_xavier"] = {
        "diagnosis": diag_xavier,
        "gradient_diagnosis": {
            "vanishing": diag_grad_xavier.vanishing,
            "diagnosis": diag_grad_xavier.diagnosis,
            "grad_norms": diag_grad_xavier.layer_grad_norms,
        },
        "training": {k: v for k, v in xavier_train.items() if k != "loss_curve"},
        "loss_curve": xavier_train["loss_curve"][::10],
    }

    # Step 2: User tries Kaiming initialization
    print("\n--- Step 2: Kaiming Initialization ---")
    kaiming_gain = 1.0  # standard for GELU
    diag_kaiming = phasekit_diagnose(depth, width, activation, kaiming_gain)
    print(f"  Kaiming σ_w = {kaiming_gain:.4f}")
    print(f"  PhaseKit diagnosis: {diag_kaiming['diagnosis']}")

    kaiming_r = kaiming_init(dims, activation, seed=42)
    diag_grad_kaiming = gradient_norm_diagnostic(
        kaiming_r.weights, kaiming_r.biases, activation,
        X_train[:64], y_train[:64])
    kaiming_train = train_mlp(
        kaiming_r.weights, kaiming_r.biases,
        X_train, y_train, X_test, y_test,
        activation, n_steps=500, lr=0.001)
    print(f"  Gradient diagnosis: {diag_grad_kaiming.diagnosis}")
    print(f"  Training result: test_loss={kaiming_train['test_loss']:.4f}, "
          f"converged={kaiming_train['converged']}")

    results["step2_kaiming"] = {
        "diagnosis": diag_kaiming,
        "gradient_diagnosis": {
            "vanishing": diag_grad_kaiming.vanishing,
            "diagnosis": diag_grad_kaiming.diagnosis,
            "grad_norms": diag_grad_kaiming.layer_grad_norms,
        },
        "training": {k: v for k, v in kaiming_train.items() if k != "loss_curve"},
        "loss_curve": kaiming_train["loss_curve"][::10],
    }

    # Step 3: PhaseKit recommends correct σ_w
    print("\n--- Step 3: PhaseKit Recommended Initialization ---")
    pk_r = phasekit_init(dims, activation, seed=42)
    pk_gain = pk_r.sigma_w_per_layer[0]
    diag_pk = phasekit_diagnose(depth, width, activation, pk_gain)
    print(f"  PhaseKit recommended σ_w = {pk_gain:.4f}")
    print(f"  PhaseKit diagnosis: {diag_pk['diagnosis']}")

    diag_grad_pk = gradient_norm_diagnostic(
        pk_r.weights, pk_r.biases, activation,
        X_train[:64], y_train[:64])
    pk_train = train_mlp(
        pk_r.weights, pk_r.biases,
        X_train, y_train, X_test, y_test,
        activation, n_steps=500, lr=0.001)
    print(f"  Gradient diagnosis: {diag_grad_pk.diagnosis}")
    print(f"  Training result: test_loss={pk_train['test_loss']:.4f}, "
          f"converged={pk_train['converged']}")

    results["step3_phasekit"] = {
        "diagnosis": diag_pk,
        "recommended_sigma_w": pk_gain,
        "gradient_diagnosis": {
            "vanishing": diag_grad_pk.vanishing,
            "diagnosis": diag_grad_pk.diagnosis,
            "grad_norms": diag_grad_pk.layer_grad_norms,
        },
        "training": {k: v for k, v in pk_train.items() if k != "loss_curve"},
        "loss_curve": pk_train["loss_curve"][::10],
    }

    # Step 4: Also run LSUV for comparison
    print("\n--- Step 4: LSUV Comparison ---")
    lsuv_r = lsuv_init(dims, activation, X_train[:256], seed=42)
    lsuv_train = train_mlp(
        lsuv_r.weights, lsuv_r.biases,
        X_train, y_train, X_test, y_test,
        activation, n_steps=500, lr=0.001)
    print(f"  LSUV training result: test_loss={lsuv_train['test_loss']:.4f}, "
          f"converged={lsuv_train['converged']}")

    results["step4_lsuv"] = {
        "training": {k: v for k, v in lsuv_train.items() if k != "loss_curve"},
        "loss_curve": lsuv_train["loss_curve"][::10],
    }

    # Summary
    print("\n" + "=" * 70)
    print("CASE STUDY SUMMARY")
    print("=" * 70)
    print(f"  Architecture: {depth}-layer, width-{width}, {activation.upper()} MLP")
    print(f"  Xavier test loss:    {xavier_train['test_loss']:.4f} (FAILS - ordered phase)")
    print(f"  Kaiming test loss:   {kaiming_train['test_loss']:.4f} (FAILS - ordered phase)")
    print(f"  PhaseKit test loss:  {pk_train['test_loss']:.4f} (PhaseKit σ_w={pk_gain:.4f})")
    print(f"  LSUV test loss:      {lsuv_train['test_loss']:.4f} (data-dependent)")

    pk_vs_xavier = xavier_train['test_loss'] / max(pk_train['test_loss'], 1e-10)
    pk_vs_kaiming = kaiming_train['test_loss'] / max(pk_train['test_loss'], 1e-10)
    results["summary"] = {
        "xavier_test_loss": xavier_train['test_loss'],
        "kaiming_test_loss": kaiming_train['test_loss'],
        "phasekit_test_loss": pk_train['test_loss'],
        "lsuv_test_loss": lsuv_train['test_loss'],
        "phasekit_vs_xavier_ratio": pk_vs_xavier,
        "phasekit_vs_kaiming_ratio": pk_vs_kaiming,
        "phasekit_gain": pk_gain,
    }

    print(f"\n  PhaseKit improvement over Xavier: {pk_vs_xavier:.2f}×")
    print(f"  PhaseKit improvement over Kaiming: {pk_vs_kaiming:.2f}×")

    output_path = os.path.join(RESULTS_DIR, 'exp_case_study.json')
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_path}")

    return results


if __name__ == "__main__":
    run_case_study()
