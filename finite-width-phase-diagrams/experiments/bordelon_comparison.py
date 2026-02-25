#!/usr/bin/env python3
"""
Comparison with Bordelon & Pehlevan 2024 predictions.

Implements their key prediction: the feature learning transition occurs when
the effective learning rate γ exceeds a critical value determined by the
spectrum of the infinite-width kernel and the task-model alignment.

Their prediction:
  γ* ∝ 1 / (λ_max · P_target)
where λ_max is the top NTK eigenvalue and P_target captures the alignment
between the target function and the kernel eigenvectors.

We compare this with our finite-width corrected prediction and show where
the 1/N corrections add value.
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation'))

from src.kernel_engine.ntk import AnalyticNTK
from src.corrections.finite_width import FiniteWidthCorrector
from src.phase_mapper.gamma_star import PhaseBoundaryPredictor
from src.corrections.trace_normalized import TraceNormalizedCorrector

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
os.makedirs(DATA_DIR, exist_ok=True)


class MLP:
    """MLP for NTK experiments."""
    def __init__(self, dims, seed=42, init_scale=1.0):
        rng = np.random.RandomState(seed)
        self.dims = dims
        self.L = len(dims) - 1
        self.weights = []
        self.param_slices = []
        idx = 0
        for i in range(self.L):
            fan_in = dims[i]
            n_params = dims[i] * dims[i+1]
            W = rng.randn(dims[i], dims[i+1]) * init_scale / np.sqrt(fan_in)
            self.weights.append(W)
            self.param_slices.append((idx, idx + n_params))
            idx += n_params
        self.n_params = idx

    def forward(self, X):
        h = X
        self.pre_activations = [X]
        self.post_activations = [X]
        for i in range(self.L):
            z = h @ self.weights[i]
            self.pre_activations.append(z)
            if i < self.L - 1:
                h = np.maximum(z, 0)
            else:
                h = z
            self.post_activations.append(h)
        return h

    def compute_jacobian(self, X):
        n = X.shape[0]
        self.forward(X)
        J = np.zeros((n, self.n_params))
        delta = np.ones((n, self.dims[-1]))
        for l in range(self.L - 1, -1, -1):
            h_prev = self.post_activations[l]
            s, e = self.param_slices[l]
            for i in range(n):
                J[i, s:e] = np.outer(h_prev[i], delta[i]).ravel()
            if l > 0:
                delta = delta @ self.weights[l].T
                relu_mask = (self.pre_activations[l] > 0).astype(float)
                delta = delta * relu_mask
        return J

    def compute_ntk(self, X):
        J = self.compute_jacobian(X)
        return J @ J.T

    def train_step(self, X, y, lr):
        pred = self.forward(X).flatten()
        residual = pred - y
        n = len(y)
        delta = residual.reshape(-1, 1) / n
        grads = []
        for l in range(self.L - 1, -1, -1):
            h_prev = self.post_activations[l]
            grad_W = h_prev.T @ delta
            grads.insert(0, grad_W)
            if l > 0:
                delta = delta @ self.weights[l].T
                relu_mask = (self.pre_activations[l] > 0).astype(float)
                delta = delta * relu_mask
        for i in range(self.L):
            self.weights[i] -= lr * grads[i]
        return 0.5 * np.mean(residual ** 2)


def bordelon_pehlevan_prediction(K_inf, y, n_samples):
    """Compute Bordelon & Pehlevan 2024 prediction for γ*.

    Their prediction: the feature learning regime activates when the
    effective coupling exceeds a threshold determined by the kernel-target
    alignment.

    γ*_BP = 1 / (λ_max · P_1)

    where P_1 = (y^T v_1)^2 / ||y||^2 is the projection of the target
    onto the top kernel eigenvector, and λ_max is the top eigenvalue.

    Parameters
    ----------
    K_inf : NDArray
        Infinite-width NTK, shape (n, n).
    y : NDArray
        Target labels, shape (n,).
    n_samples : int
        Number of data points.

    Returns
    -------
    dict with prediction details
    """
    eigs, V = np.linalg.eigh(K_inf)
    idx = np.argsort(eigs)[::-1]
    eigs = eigs[idx]
    V = V[:, idx]

    lambda_max = float(eigs[0])

    # Target alignment with top eigenvectors
    y_norm = np.linalg.norm(y)
    projections = (V.T @ y) ** 2 / max(y_norm ** 2, 1e-15)

    # P_1: projection onto top eigenspace
    P_1 = float(projections[0])

    # Their threshold (simplified)
    if lambda_max * P_1 > 1e-15:
        gamma_star_bp = 1.0 / (lambda_max * P_1 / n_samples)
    else:
        gamma_star_bp = float('inf')

    # Extended prediction using top-k eigenvalues
    k = min(5, len(eigs))
    P_k = float(np.sum(projections[:k]))
    lambda_eff = float(np.sum(eigs[:k] * projections[:k]) / max(P_k, 1e-15))

    if lambda_eff * P_k > 1e-15:
        gamma_star_bp_extended = 1.0 / (lambda_eff * P_k / n_samples)
    else:
        gamma_star_bp_extended = gamma_star_bp

    return {
        'gamma_star': gamma_star_bp,
        'gamma_star_extended': gamma_star_bp_extended,
        'lambda_max': lambda_max,
        'P_1': P_1,
        'P_k': P_k,
        'lambda_eff': lambda_eff,
        'top_eigenvalues': eigs[:k].tolist(),
        'top_projections': projections[:k].tolist(),
    }


def run_comparison():
    """Run head-to-head comparison between our method and BP2024."""
    print("=" * 70)
    print("COMPARISON WITH BORDELON & PEHLEVAN 2024")
    print("=" * 70)

    input_dim = 16
    n_samples = 30
    depths = [2, 3]
    n_seeds = 3
    T_values = [50, 100, 200, 500]

    rng = np.random.RandomState(42)
    X = rng.randn(n_samples, input_dim)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    y = rng.randn(n_samples)

    results = {}
    predictor = PhaseBoundaryPredictor()

    for depth in depths:
        print(f"\n  Depth {depth}:")
        width_test = 128

        # Infinite-width NTK
        antk = AnalyticNTK(depth=depth + 1, activation='relu')
        K_inf = antk.compute_ntk(X)

        # BP2024 prediction
        bp_result = bordelon_pehlevan_prediction(K_inf, y, n_samples)
        print(f"    BP2024 γ* = {bp_result['gamma_star']:.6f}")
        print(f"    BP2024 γ* (extended) = {bp_result['gamma_star_extended']:.6f}")

        # Our prediction: fit corrections from multiple widths
        cal_widths = [64, 128, 256, 512]
        cal_ntks = []
        for w in cal_widths:
            seed_ntks = []
            for seed in range(n_seeds):
                net = MLP([input_dim] + [w]*depth + [1], seed=seed+100)
                seed_ntks.append(net.compute_ntk(X))
            cal_ntks.append(np.mean(seed_ntks, axis=0))

        corrector = FiniteWidthCorrector(output_dim=1, min_widths=3)
        correction = corrector.compute_corrections_regression(
            np.array(cal_ntks), cal_widths
        )

        # Our predictions at different T
        our_predictions = {}
        for T in T_values:
            gs_result = predictor.predict_gamma_star(
                correction.theta_0, correction.theta_1, T, width_test
            )
            our_predictions[T] = gs_result.gamma_star

        print(f"    Our γ* (T=100): {our_predictions.get(100, 'N/A'):.6f}")
        print(f"    Our γ* (T=500): {our_predictions.get(500, 'N/A'):.6f}")

        # Empirical measurement
        lr_range = np.logspace(-4.0, 0.0, 20)
        empirical_results = {}

        for T in T_values:
            for lr in lr_range:
                seed_drifts = []
                for seed in range(n_seeds):
                    net = MLP([input_dim] + [width_test]*depth + [1], seed=seed+100)
                    K0 = net.compute_ntk(X)
                    K0_norm = np.linalg.norm(K0, 'fro')
                    for step in range(T):
                        net.train_step(X, y, lr)
                    Kt = net.compute_ntk(X)
                    drift = np.linalg.norm(Kt - K0, 'fro') / K0_norm
                    seed_drifts.append(float(drift))

                gamma = lr / width_test
                if T not in empirical_results:
                    empirical_results[T] = []
                empirical_results[T].append({
                    'lr': float(lr),
                    'gamma': float(gamma),
                    'mean_drift': float(np.mean(seed_drifts)),
                })

        # Find empirical γ*
        empirical_gamma_stars = {}
        for T, lr_res in empirical_results.items():
            for k in range(len(lr_res) - 1):
                if lr_res[k]['mean_drift'] < 0.1 and lr_res[k+1]['mean_drift'] >= 0.1:
                    t_interp = (0.1 - lr_res[k]['mean_drift']) / (
                        lr_res[k+1]['mean_drift'] - lr_res[k]['mean_drift'] + 1e-10)
                    gs_emp = float(np.exp(
                        np.log(lr_res[k]['gamma'] + 1e-10) +
                        t_interp * (np.log(lr_res[k+1]['gamma'] + 1e-10) -
                                    np.log(lr_res[k]['gamma'] + 1e-10))))
                    empirical_gamma_stars[T] = gs_emp
                    break

        # Compare predictions
        comparison = []
        for T in T_values:
            entry = {
                'T': T,
                'bp2024_gamma_star': bp_result['gamma_star'],
                'bp2024_gamma_star_extended': bp_result['gamma_star_extended'],
                'our_gamma_star': our_predictions.get(T, float('inf')),
                'empirical_gamma_star': empirical_gamma_stars.get(T),
            }

            if entry['empirical_gamma_star'] is not None:
                emp = entry['empirical_gamma_star']
                bp_err = abs(bp_result['gamma_star'] - emp) / emp
                our_err = abs(our_predictions.get(T, float('inf')) - emp) / emp
                entry['bp2024_relative_error'] = float(bp_err)
                entry['our_relative_error'] = float(our_err)
                entry['improvement_factor'] = float(bp_err / max(our_err, 1e-10))
                print(f"    T={T}: emp={emp:.6f}, BP_err={bp_err:.2f}, ours_err={our_err:.2f}, "
                      f"improvement={entry['improvement_factor']:.2f}x")

            comparison.append(entry)

        results[f'depth_{depth}'] = {
            'depth': depth,
            'width': width_test,
            'bp2024': bp_result,
            'our_correction_ratio': float(
                np.linalg.norm(correction.theta_1) /
                max(np.linalg.norm(correction.theta_0), 1e-10)
            ),
            'comparison': comparison,
        }

    results['metadata'] = {
        'experiment': 'bordelon_pehlevan_comparison',
        'input_dim': input_dim,
        'n_samples': n_samples,
        'n_seeds': n_seeds,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = os.path.join(DATA_DIR, 'exp_bordelon_comparison.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    return results


if __name__ == '__main__':
    run_comparison()
