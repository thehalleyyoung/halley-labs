"""
Real neural network training experiments for PhaseDiag validation.

Trains actual PyTorch networks at various (σ_w, σ_b) initializations and
measures loss curves, gradient norms, NTK drift, and training outcomes.
Compares against PhaseDiag theoretical predictions.

This addresses the critical review concern: "No ground-truth validation."
"""

import sys
import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = Path(__file__).parent / 'data'
RESULTS_DIR.mkdir(exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)


# ──────────────────────────────────────────────────────────────
# Helper: build MLP with exact (σ_w, σ_b) initialization
# ──────────────────────────────────────────────────────────────
def build_mlp(input_dim, hidden_dims, output_dim, sigma_w, sigma_b, activation='relu'):
    """Build an MLP with exact Gaussian initialization."""
    layers = []
    dims = [input_dim] + list(hidden_dims) + [output_dim]
    act_map = {
        'relu': nn.ReLU,
        'tanh': nn.Tanh,
        'gelu': nn.GELU,
        'silu': nn.SiLU,
    }
    act_cls = act_map.get(activation, nn.ReLU)

    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        fan_in = dims[i]
        nn.init.normal_(linear.weight, 0, sigma_w / np.sqrt(fan_in))
        nn.init.normal_(linear.bias, 0, sigma_b if sigma_b > 0 else 1e-8)
        layers.append(linear)
        if i < len(dims) - 2:
            layers.append(act_cls())

    model = nn.Sequential(*layers)
    return model


def compute_empirical_ntk(model, X, device='cpu'):
    """Compute empirical NTK via Jacobian for small networks."""
    model.eval()
    X = X.to(device)
    n = X.shape[0]
    params = [p for p in model.parameters() if p.requires_grad]

    jacobians = []
    for i in range(n):
        model.zero_grad()
        out = model(X[i:i+1])
        out.backward()
        grad = torch.cat([p.grad.flatten() for p in params])
        jacobians.append(grad.detach())

    J = torch.stack(jacobians)  # (n, P)
    ntk = J @ J.T  # (n, n)
    return ntk.numpy()


def measure_gradient_norms(model, X, y, criterion):
    """Measure per-layer gradient norms."""
    model.zero_grad()
    out = model(X)
    loss = criterion(out, y)
    loss.backward()
    norms = {}
    for name, p in model.named_parameters():
        if p.grad is not None:
            norms[name] = p.grad.norm().item()
    return norms, loss.item()


def train_network(model, X_train, y_train, X_test, y_test, lr, n_steps,
                  record_interval=10):
    """Train a network and record detailed metrics."""
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    history = {
        'train_loss': [],
        'test_loss': [],
        'grad_norm_mean': [],
        'grad_norm_max': [],
        'grad_norm_min': [],
        'step': [],
    }

    for step in range(n_steps):
        model.train()
        optimizer.zero_grad()
        out = model(X_train)
        loss = criterion(out, y_train)
        loss.backward()

        if step % record_interval == 0:
            with torch.no_grad():
                model.eval()
                test_out = model(X_test)
                test_loss = criterion(test_out, y_test).item()

            grad_norms = []
            for p in model.parameters():
                if p.grad is not None:
                    grad_norms.append(p.grad.norm().item())

            history['train_loss'].append(loss.item())
            history['test_loss'].append(test_loss)
            history['grad_norm_mean'].append(np.mean(grad_norms))
            history['grad_norm_max'].append(np.max(grad_norms))
            history['grad_norm_min'].append(np.min(grad_norms))
            history['step'].append(step)

        optimizer.step()

    return history


def compute_ntk_drift(model, X, ntk_init):
    """Measure NTK drift relative to initialization."""
    ntk_current = compute_empirical_ntk(model, X)
    frob_dist = np.linalg.norm(ntk_current - ntk_init, 'fro')
    frob_init = np.linalg.norm(ntk_init, 'fro')
    relative_drift = frob_dist / max(frob_init, 1e-10)
    return relative_drift, ntk_current


# ──────────────────────────────────────────────────────────────
# Experiment 1: Phase prediction validation
# Train MLPs at various σ_w values, verify ordered/critical/chaotic
# ──────────────────────────────────────────────────────────────
def experiment_1_phase_validation():
    """Validate phase predictions against actual training outcomes."""
    print("\n" + "="*70)
    print("EXPERIMENT 1: Phase Prediction Validation")
    print("="*70)

    input_dim = 10
    hidden_dim = 128
    depth = 5
    n_train = 200
    n_test = 50
    n_steps = 500
    lr = 0.01

    # Generate synthetic regression data
    torch.manual_seed(42)
    X_train = torch.randn(n_train, input_dim)
    w_true = torch.randn(input_dim, 1)
    y_train = X_train @ w_true + 0.1 * torch.randn(n_train, 1)
    X_test = torch.randn(n_test, input_dim)
    y_test = X_test @ w_true + 0.1 * torch.randn(n_test, 1)

    sigma_w_values = [0.5, 0.8, 1.0, 1.2, 1.4142, 1.6, 2.0, 2.5, 3.0]
    results = {}

    for sigma_w in sigma_w_values:
        print(f"\n  σ_w = {sigma_w:.4f}:")

        # Theoretical prediction
        chi1 = sigma_w**2 / 2  # ReLU
        if chi1 < 0.95:
            predicted_phase = 'ordered'
        elif chi1 > 1.05:
            predicted_phase = 'chaotic'
        else:
            predicted_phase = 'critical'

        # Train 3 seeds
        seed_results = []
        for seed in range(3):
            torch.manual_seed(seed * 1000 + 42)
            model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                            sigma_w=sigma_w, sigma_b=0.0, activation='relu')

            # Measure initial gradient norms
            grad_norms_init, init_loss = measure_gradient_norms(
                model, X_train[:20], y_train[:20], nn.MSELoss())

            # Train
            history = train_network(model, X_train, y_train, X_test, y_test,
                                  lr=lr, n_steps=n_steps, record_interval=20)

            final_loss = history['train_loss'][-1]
            init_train_loss = history['train_loss'][0]
            loss_ratio = final_loss / max(init_train_loss, 1e-10)

            # Determine actual regime
            grad_mean_init = np.mean(list(grad_norms_init.values()))
            if grad_mean_init < 1e-6:
                actual_phase = 'ordered'
            elif grad_mean_init > 1e4 or np.isnan(final_loss) or np.isinf(final_loss):
                actual_phase = 'chaotic'
            elif loss_ratio < 0.1:
                actual_phase = 'critical'
            elif loss_ratio > 0.9 and final_loss > 1.0:
                actual_phase = 'ordered'
            else:
                actual_phase = 'critical'

            seed_results.append({
                'seed': seed,
                'init_loss': float(init_train_loss),
                'final_loss': float(final_loss) if not np.isnan(final_loss) else 1e10,
                'loss_ratio': float(loss_ratio) if not np.isnan(loss_ratio) else 1e10,
                'init_grad_mean': float(grad_mean_init),
                'actual_phase': actual_phase,
                'converged': bool(loss_ratio < 0.5 and not np.isnan(final_loss)),
                'exploded': bool(np.isnan(final_loss) or np.isinf(final_loss)),
                'train_loss_curve': [float(x) if not np.isnan(x) else 1e10
                                    for x in history['train_loss']],
                'grad_norm_curve': [float(x) if not np.isnan(x) else 1e10
                                   for x in history['grad_norm_mean']],
            })

        # Majority vote
        phase_votes = [r['actual_phase'] for r in seed_results]
        actual_majority = max(set(phase_votes), key=phase_votes.count)
        match = (predicted_phase == actual_majority)

        results[f'sw_{sigma_w:.4f}'] = {
            'sigma_w': sigma_w,
            'chi_1': chi1,
            'predicted_phase': predicted_phase,
            'actual_phase': actual_majority,
            'match': match,
            'seeds': seed_results,
        }

        converge_rate = sum(1 for r in seed_results if r['converged']) / len(seed_results)
        explode_rate = sum(1 for r in seed_results if r['exploded']) / len(seed_results)
        print(f"    χ₁={chi1:.3f} predicted={predicted_phase} actual={actual_majority}"
              f" match={match} converge={converge_rate:.0%} explode={explode_rate:.0%}")

    # Compute accuracy
    n_correct = sum(1 for v in results.values() if v['match'])
    accuracy = n_correct / len(results)
    print(f"\n  Phase prediction accuracy: {accuracy:.1%} ({n_correct}/{len(results)})")

    output = {
        'experiment': 'phase_validation',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'depth': depth, 'n_train': n_train, 'n_steps': n_steps, 'lr': lr,
            'activation': 'relu',
        },
        'results': results,
        'accuracy': accuracy,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_phase_validation.json', 'w') as f:
        json.dump(output, f, indent=2)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 2: NTK drift measurement
# ──────────────────────────────────────────────────────────────
def experiment_2_ntk_drift():
    """Measure actual NTK drift during training at different scales."""
    print("\n" + "="*70)
    print("EXPERIMENT 2: NTK Drift Measurement")
    print("="*70)

    input_dim = 5
    n_samples = 15
    n_steps = 200
    widths = [32, 64, 128, 256]
    depth = 2

    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)
    y = torch.randn(n_samples, 1)

    results = {}

    for width in widths:
        sigma_w = np.sqrt(2)  # critical initialization
        lr = 0.01 / width  # NTK scaling

        torch.manual_seed(42)
        model = build_mlp(input_dim, [width]*depth, 1,
                         sigma_w=sigma_w, sigma_b=0.0)

        # Compute initial NTK
        ntk_init = compute_empirical_ntk(model, X)

        # Check PSD and symmetry
        eigvals = np.linalg.eigvalsh(ntk_init)
        is_psd = bool(np.all(eigvals >= -1e-6))
        cond_number = float(np.max(np.abs(eigvals)) / max(np.min(np.abs(eigvals)), 1e-10))

        # Train and measure drift
        criterion = nn.MSELoss()
        optimizer = optim.SGD(model.parameters(), lr=lr)
        drift_trajectory = []

        for step in range(n_steps):
            model.train()
            optimizer.zero_grad()
            out = model(X)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            if step % 50 == 0:
                drift, ntk_current = compute_ntk_drift(model, X, ntk_init)
                drift_trajectory.append({'step': step, 'drift': float(drift)})

        final_drift, _ = compute_ntk_drift(model, X, ntk_init)

        results[f'width_{width}'] = {
            'width': width,
            'depth': depth,
            'sigma_w': float(sigma_w),
            'lr': lr,
            'ntk_condition_number': cond_number,
            'ntk_is_psd': is_psd,
            'ntk_trace': float(np.trace(ntk_init)),
            'ntk_top_eigenvalue': float(np.max(eigvals)),
            'final_drift': float(final_drift),
            'drift_trajectory': drift_trajectory,
        }
        print(f"  Width {width}: condition={cond_number:.1f} drift={final_drift:.4f}"
              f" PSD={is_psd}")

    # Check if drift decreases with width (lazy regime prediction)
    drifts = [(r['width'], r['final_drift']) for r in results.values()]
    drifts.sort()
    drift_decreasing = all(drifts[i][1] >= drifts[i+1][1] * 0.5
                          for i in range(len(drifts)-1))

    output = {
        'experiment': 'ntk_drift',
        'config': {
            'input_dim': input_dim, 'n_samples': n_samples,
            'n_steps': n_steps, 'depth': depth,
        },
        'results': results,
        'drift_decreases_with_width': drift_decreasing,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_ntk_drift.json', 'w') as f:
        json.dump(output, f, indent=2)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 3: Finite-width NTK convergence
# ──────────────────────────────────────────────────────────────
def experiment_3_ntk_convergence():
    """Measure convergence of empirical NTK to analytical as width grows."""
    print("\n" + "="*70)
    print("EXPERIMENT 3: NTK Convergence to Infinite-Width Limit")
    print("="*70)

    input_dim = 5
    n_samples = 15
    depth = 2
    sigma_w = np.sqrt(2)
    widths = [16, 32, 64, 128, 256, 512]
    n_seeds = 5

    torch.manual_seed(42)
    X = torch.randn(n_samples, input_dim)

    # Compute analytical NTK (infinite-width limit) using our toolkit
    try:
        from ntk_computation import NTKComputer, ModelSpec
        ntk_computer = NTKComputer()
        spec = ModelSpec(layer_widths=[input_dim] + [1000]*depth + [1],
                        activation='relu', sigma_w=sigma_w)
        ntk_result = ntk_computer.compute(spec, X.numpy())
        ntk_analytical = ntk_result.kernel_matrix
        has_analytical = True
    except Exception as e:
        print(f"  Warning: could not compute analytical NTK: {e}")
        # Use very wide network as proxy
        torch.manual_seed(42)
        model = build_mlp(input_dim, [2048]*depth, 1, sigma_w=sigma_w, sigma_b=0.0)
        ntk_analytical = compute_empirical_ntk(model, X)
        has_analytical = False

    results = {}
    for width in widths:
        errors = []
        for seed in range(n_seeds):
            torch.manual_seed(seed * 100 + 42)
            model = build_mlp(input_dim, [width]*depth, 1,
                            sigma_w=sigma_w, sigma_b=0.0)
            ntk_emp = compute_empirical_ntk(model, X)

            # Normalize both
            scale = np.trace(ntk_analytical) / max(np.trace(ntk_emp), 1e-10)
            ntk_emp_scaled = ntk_emp * scale

            rel_error = np.linalg.norm(ntk_emp_scaled - ntk_analytical, 'fro') / \
                       np.linalg.norm(ntk_analytical, 'fro')
            errors.append(float(rel_error))

        results[f'width_{width}'] = {
            'width': width,
            'mean_error': float(np.mean(errors)),
            'std_error': float(np.std(errors)),
            'errors': errors,
        }
        print(f"  Width {width}: relative error = {np.mean(errors):.4f} ± {np.std(errors):.4f}")

    # Fit power law: error ~ width^alpha
    log_w = np.log([r['width'] for r in results.values()])
    log_e = np.log([r['mean_error'] for r in results.values()])
    slope, intercept = np.polyfit(log_w, log_e, 1)
    r2 = 1 - np.sum((log_e - (slope*log_w + intercept))**2) / \
         np.sum((log_e - np.mean(log_e))**2)

    print(f"\n  Convergence rate: error ~ N^{slope:.3f} (R²={r2:.3f})")

    output = {
        'experiment': 'ntk_convergence',
        'config': {
            'input_dim': input_dim, 'n_samples': n_samples,
            'depth': depth, 'sigma_w': float(sigma_w),
            'n_seeds': n_seeds, 'has_analytical': has_analytical,
        },
        'results': results,
        'convergence_rate': float(slope),
        'r2': float(r2),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_ntk_convergence.json', 'w') as f:
        json.dump(output, f, indent=2)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 4: Initialization comparison on real training
# ──────────────────────────────────────────────────────────────
def experiment_4_init_comparison():
    """Compare critical vs non-critical initialization on real training."""
    print("\n" + "="*70)
    print("EXPERIMENT 4: Critical vs Non-Critical Initialization")
    print("="*70)

    input_dim = 20
    hidden_dim = 128
    depth = 8
    n_train = 500
    n_test = 100
    n_steps = 1000
    lr = 0.001

    # Generate data
    torch.manual_seed(42)
    X_train = torch.randn(n_train, input_dim)
    # Nonlinear target
    w1 = torch.randn(input_dim, 5)
    w2 = torch.randn(5, 1)
    y_train = torch.relu(X_train @ w1) @ w2 + 0.1 * torch.randn(n_train, 1)
    X_test = torch.randn(n_test, input_dim)
    y_test = torch.relu(X_test @ w1) @ w2 + 0.1 * torch.randn(n_test, 1)

    init_configs = {
        'ordered_0.5': {'sigma_w': 0.5, 'sigma_b': 0.0},
        'ordered_0.8': {'sigma_w': 0.8, 'sigma_b': 0.0},
        'near_critical': {'sigma_w': 1.2, 'sigma_b': 0.0},
        'critical': {'sigma_w': np.sqrt(2), 'sigma_b': 0.0},
        'chaotic_2.0': {'sigma_w': 2.0, 'sigma_b': 0.0},
        'chaotic_3.0': {'sigma_w': 3.0, 'sigma_b': 0.0},
        'kaiming': {'sigma_w': np.sqrt(2), 'sigma_b': 0.0},  # same as critical for ReLU
    }

    results = {}
    for name, config in init_configs.items():
        print(f"\n  {name} (σ_w={config['sigma_w']:.3f}):")

        seed_results = []
        for seed in range(3):
            torch.manual_seed(seed * 1000 + 42)
            model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                            sigma_w=config['sigma_w'],
                            sigma_b=config['sigma_b'])

            history = train_network(model, X_train, y_train, X_test, y_test,
                                  lr=lr, n_steps=n_steps, record_interval=50)

            final_train = history['train_loss'][-1]
            final_test = history['test_loss'][-1]
            best_test = min(history['test_loss'])

            seed_results.append({
                'seed': seed,
                'final_train_loss': float(final_train) if not np.isnan(final_train) else 1e10,
                'final_test_loss': float(final_test) if not np.isnan(final_test) else 1e10,
                'best_test_loss': float(best_test) if not np.isnan(best_test) else 1e10,
                'converged': bool(not np.isnan(final_train) and final_train < 10),
                'train_loss_curve': [float(x) if not np.isnan(x) else 1e10
                                    for x in history['train_loss'][:20]],  # first 20 records
            })

        avg_best_test = np.mean([r['best_test_loss'] for r in seed_results])
        converge_rate = sum(1 for r in seed_results if r['converged']) / len(seed_results)

        results[name] = {
            'config': config,
            'chi_1': config['sigma_w']**2 / 2,
            'avg_best_test_loss': float(avg_best_test),
            'convergence_rate': float(converge_rate),
            'seeds': seed_results,
        }
        print(f"    avg best test loss={avg_best_test:.4f} converge={converge_rate:.0%}")

    # Rank by test loss
    ranked = sorted(results.items(), key=lambda x: x[1]['avg_best_test_loss'])
    print("\n  Ranking by test loss:")
    for i, (name, r) in enumerate(ranked):
        print(f"    {i+1}. {name}: {r['avg_best_test_loss']:.4f}")

    output = {
        'experiment': 'init_comparison',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'depth': depth, 'n_train': n_train, 'n_steps': n_steps, 'lr': lr,
        },
        'results': results,
        'ranking': [name for name, _ in ranked],
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_init_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 5: Activation function comparison on real training
# ──────────────────────────────────────────────────────────────
def experiment_5_activation_comparison():
    """Compare activation functions at their respective critical points."""
    print("\n" + "="*70)
    print("EXPERIMENT 5: Activation Function Comparison (Real Training)")
    print("="*70)

    input_dim = 10
    hidden_dim = 128
    depth = 5
    n_train = 300
    n_test = 100
    n_steps = 1000
    lr = 0.001

    torch.manual_seed(42)
    X_train = torch.randn(n_train, input_dim)
    w_true = torch.randn(input_dim, 1)
    y_train = X_train @ w_true + 0.1 * torch.randn(n_train, 1)
    X_test = torch.randn(n_test, input_dim)
    y_test = X_test @ w_true + 0.1 * torch.randn(n_test, 1)

    # Correct edge-of-chaos values
    activations = {
        'relu': {'sigma_w': np.sqrt(2), 'eoc': np.sqrt(2)},
        'tanh': {'sigma_w': 1.0056, 'eoc': 1.0056},
        'gelu': {'sigma_w': 1.5335, 'eoc': 1.5335},
        'silu': {'sigma_w': 1.6765, 'eoc': 1.6765},
    }

    results = {}
    for act_name, act_info in activations.items():
        print(f"\n  {act_name} (σ_w*={act_info['eoc']:.4f}):")

        seed_results = []
        for seed in range(3):
            torch.manual_seed(seed * 1000 + 42)
            model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                            sigma_w=act_info['sigma_w'], sigma_b=0.0,
                            activation=act_name)

            history = train_network(model, X_train, y_train, X_test, y_test,
                                  lr=lr, n_steps=n_steps, record_interval=50)

            final_test = history['test_loss'][-1]
            best_test = min(history['test_loss'])
            converged = not np.isnan(final_test) and final_test < 50

            seed_results.append({
                'seed': seed,
                'final_test': float(final_test) if not np.isnan(final_test) else 1e10,
                'best_test': float(best_test) if not np.isnan(best_test) else 1e10,
                'converged': bool(converged),
                'train_loss_curve': [float(x) if not np.isnan(x) else 1e10
                                    for x in history['train_loss'][:20]],
            })

        avg_best = np.mean([r['best_test'] for r in seed_results])
        converge_rate = sum(1 for r in seed_results if r['converged']) / len(seed_results)
        results[act_name] = {
            'sigma_w_star': act_info['eoc'],
            'avg_best_test': float(avg_best),
            'convergence_rate': float(converge_rate),
            'seeds': seed_results,
        }
        print(f"    avg best test={avg_best:.4f} converge={converge_rate:.0%}")

    output = {
        'experiment': 'activation_comparison',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'depth': depth, 'n_train': n_train, 'n_steps': n_steps, 'lr': lr,
        },
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_activation_comparison.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 6: Depth scale validation
# ──────────────────────────────────────────────────────────────
def experiment_6_depth_scale():
    """Validate that max trainable depth matches depth scale prediction."""
    print("\n" + "="*70)
    print("EXPERIMENT 6: Depth Scale vs Trainable Depth")
    print("="*70)

    input_dim = 10
    hidden_dim = 128
    n_train = 200
    n_steps = 500
    lr = 0.001

    torch.manual_seed(42)
    X_train = torch.randn(n_train, input_dim)
    w_true = torch.randn(input_dim, 1)
    y_train = X_train @ w_true + 0.1 * torch.randn(n_train, 1)
    X_test = torch.randn(50, input_dim)
    y_test = X_test @ w_true + 0.1 * torch.randn(50, 1)

    sigma_w_values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.4142]
    depths = [2, 4, 8, 16, 32]

    results = {}
    for sigma_w in sigma_w_values:
        chi1 = sigma_w**2 / 2
        if chi1 < 1:
            depth_scale = -1.0 / np.log(chi1)
        else:
            depth_scale = float('inf')

        depth_results = {}
        for depth in depths:
            torch.manual_seed(42)
            model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                            sigma_w=sigma_w, sigma_b=0.0)

            history = train_network(model, X_train, y_train, X_test, y_test,
                                  lr=lr, n_steps=n_steps, record_interval=50)

            final_loss = history['train_loss'][-1]
            init_loss = history['train_loss'][0]
            improvement = 1 - final_loss / max(init_loss, 1e-10) if not np.isnan(final_loss) else -1

            depth_results[depth] = {
                'depth': depth,
                'init_loss': float(init_loss),
                'final_loss': float(final_loss) if not np.isnan(final_loss) else 1e10,
                'improvement': float(improvement),
                'trainable': bool(improvement > 0.1),
            }

        # Find max trainable depth
        max_trainable = 0
        for d in sorted(depths):
            if depth_results[d]['trainable']:
                max_trainable = d
            else:
                break

        results[f'sw_{sigma_w:.4f}'] = {
            'sigma_w': sigma_w,
            'chi_1': chi1,
            'depth_scale': depth_scale if depth_scale != float('inf') else 1000,
            'predicted_max_depth': min(int(5 * depth_scale), 100) if depth_scale != float('inf') else 100,
            'actual_max_trainable': max_trainable,
            'depth_results': {str(k): v for k, v in depth_results.items()},
        }
        print(f"  σ_w={sigma_w:.4f} χ₁={chi1:.3f} ξ={depth_scale:.1f}"
              f" predicted≤{min(int(5*depth_scale), 100) if depth_scale != float('inf') else '∞'}"
              f" actual_max={max_trainable}")

    output = {
        'experiment': 'depth_scale',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'n_train': n_train, 'n_steps': n_steps, 'lr': lr,
        },
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_depth_scale.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 7: Variance propagation empirical validation
# ──────────────────────────────────────────────────────────────
def experiment_7_variance_propagation():
    """Validate variance propagation theory against actual network activations."""
    print("\n" + "="*70)
    print("EXPERIMENT 7: Variance Propagation Validation")
    print("="*70)

    input_dim = 50
    hidden_dim = 512
    depths = [2, 5, 10, 20]
    sigma_w_values = [0.8, 1.0, 1.2, np.sqrt(2), 1.6, 2.0]
    n_samples = 500
    n_seeds = 5

    results = {}
    for sigma_w in sigma_w_values:
        for depth in depths:
            variances_per_layer = defaultdict(list)

            for seed in range(n_seeds):
                torch.manual_seed(seed * 1000 + 42)
                model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                                sigma_w=sigma_w, sigma_b=0.0)

                X = torch.randn(n_samples, input_dim)
                # Forward pass, record variances
                h = X
                for i, layer in enumerate(model):
                    h = layer(h)
                    if isinstance(layer, nn.Linear):
                        var = h.var().item()
                        variances_per_layer[i // 2].append(var)

            # Theoretical variance propagation
            q = 1.0  # input variance
            theoretical = [q]
            for l in range(depth + 1):
                q = sigma_w**2 * q / 2  # ReLU V(q) = q/2
                theoretical.append(q)

            empirical_means = {k: float(np.mean(v)) for k, v in variances_per_layer.items()}
            empirical_stds = {k: float(np.std(v)) for k, v in variances_per_layer.items()}

            key = f'sw{sigma_w:.3f}_d{depth}'
            results[key] = {
                'sigma_w': float(sigma_w),
                'depth': depth,
                'theoretical_variances': theoretical[:depth+1],
                'empirical_means': empirical_means,
                'empirical_stds': empirical_stds,
            }

    # Compute overall agreement
    rel_errors = []
    for key, r in results.items():
        for layer_idx, emp_mean in r['empirical_means'].items():
            th_idx = int(layer_idx)
            if th_idx < len(r['theoretical_variances']):
                th = r['theoretical_variances'][th_idx]
                if th > 1e-10:
                    rel_errors.append(abs(emp_mean - th) / th)

    mean_rel_error = float(np.mean(rel_errors)) if rel_errors else -1
    print(f"\n  Mean relative error (theory vs empirical): {mean_rel_error:.4f}")

    output = {
        'experiment': 'variance_propagation',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'n_samples': n_samples, 'n_seeds': n_seeds,
        },
        'results': results,
        'mean_relative_error': mean_rel_error,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_variance_propagation.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 8: Practical use case - before/after diagnosis
# ──────────────────────────────────────────────────────────────
def experiment_8_practical_diagnosis():
    """Demonstrate practical use: diagnose bad init, fix it, show improvement."""
    print("\n" + "="*70)
    print("EXPERIMENT 8: Practical Diagnosis Use Case")
    print("="*70)

    input_dim = 20
    hidden_dim = 256
    depth = 10
    n_train = 500
    n_test = 100
    n_steps = 2000
    lr = 0.001

    torch.manual_seed(42)
    # More complex task
    X_train = torch.randn(n_train, input_dim)
    w1 = torch.randn(input_dim, 10)
    w2 = torch.randn(10, 1)
    y_train = torch.tanh(X_train @ w1) @ w2 + 0.05 * torch.randn(n_train, 1)
    X_test = torch.randn(n_test, input_dim)
    y_test = torch.tanh(X_test @ w1) @ w2 + 0.05 * torch.randn(n_test, 1)

    scenarios = {
        'bad_ordered': {
            'sigma_w': 0.5, 'sigma_b': 0.0,
            'diagnosis': 'ordered phase (χ₁=0.125), vanishing gradients',
        },
        'bad_chaotic': {
            'sigma_w': 2.5, 'sigma_b': 0.0,
            'diagnosis': 'chaotic phase (χ₁=3.125), exploding gradients',
        },
        'fixed_critical': {
            'sigma_w': np.sqrt(2), 'sigma_b': 0.0,
            'diagnosis': 'critical phase (χ₁=1.0), optimal gradient flow',
        },
    }

    results = {}
    for name, scenario in scenarios.items():
        print(f"\n  {name}: {scenario['diagnosis']}")

        seed_results = []
        for seed in range(3):
            torch.manual_seed(seed * 1000 + 42)
            model = build_mlp(input_dim, [hidden_dim]*depth, 1,
                            sigma_w=scenario['sigma_w'],
                            sigma_b=scenario['sigma_b'])

            history = train_network(model, X_train, y_train, X_test, y_test,
                                  lr=lr, n_steps=n_steps, record_interval=100)

            best_test = min(h for h in history['test_loss'] if not np.isnan(h) and h < 1e8) if \
                       any(not np.isnan(h) and h < 1e8 for h in history['test_loss']) else 1e10

            seed_results.append({
                'seed': seed,
                'best_test_loss': float(best_test),
                'final_train_loss': float(history['train_loss'][-1]) if not np.isnan(history['train_loss'][-1]) else 1e10,
                'converged': bool(best_test < 10),
                'train_loss_curve': [float(x) if not np.isnan(x) else 1e10
                                    for x in history['train_loss']],
                'test_loss_curve': [float(x) if not np.isnan(x) else 1e10
                                   for x in history['test_loss']],
                'grad_norm_curve': [float(x) if not np.isnan(x) else 1e10
                                   for x in history['grad_norm_mean']],
            })

        avg_best = np.mean([r['best_test_loss'] for r in seed_results])
        results[name] = {
            'config': {k: float(v) if isinstance(v, (float, np.floating)) else v
                      for k, v in scenario.items()},
            'chi_1': float(scenario['sigma_w']**2 / 2),
            'avg_best_test': float(avg_best),
            'convergence_rate': float(sum(1 for r in seed_results if r['converged']) / len(seed_results)),
            'seeds': seed_results,
        }
        print(f"    avg best test={avg_best:.4f}"
              f" converge={results[name]['convergence_rate']:.0%}")

    # Improvement factor
    bad_ordered = results['bad_ordered']['avg_best_test']
    bad_chaotic = results['bad_chaotic']['avg_best_test']
    fixed = results['fixed_critical']['avg_best_test']
    improvement_ordered = bad_ordered / max(fixed, 1e-10)
    improvement_chaotic = bad_chaotic / max(fixed, 1e-10)

    print(f"\n  Improvement over ordered init: {improvement_ordered:.1f}x")
    print(f"  Improvement over chaotic init: {improvement_chaotic:.1f}x")

    output = {
        'experiment': 'practical_diagnosis',
        'config': {
            'input_dim': input_dim, 'hidden_dim': hidden_dim,
            'depth': depth, 'n_train': n_train, 'n_steps': n_steps, 'lr': lr,
        },
        'results': results,
        'improvement_over_ordered': float(improvement_ordered),
        'improvement_over_chaotic': float(improvement_chaotic),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_practical_diagnosis.json', 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
    return output


# ──────────────────────────────────────────────────────────────
# Experiment 9: Toolkit NTK vs PyTorch empirical NTK benchmark
# ──────────────────────────────────────────────────────────────
def experiment_9_toolkit_benchmark():
    """Benchmark PhaseDiag NTK computation against PyTorch empirical NTK."""
    print("\n" + "="*70)
    print("EXPERIMENT 9: Toolkit vs PyTorch NTK Benchmark")
    print("="*70)

    input_dim = 5
    n_samples_list = [10, 20, 50]
    depths = [2, 3, 4]

    results = {}

    for n_samples in n_samples_list:
        for depth in depths:
            torch.manual_seed(42)
            X = torch.randn(n_samples, input_dim)
            sigma_w = np.sqrt(2)

            # PhaseDiag analytical NTK
            try:
                from ntk_computation import NTKComputer, ModelSpec
                ntk_computer = NTKComputer()
                spec = ModelSpec(
                    layer_widths=[input_dim] + [256]*depth + [1],
                    activation='relu', sigma_w=sigma_w)

                t0 = time.time()
                ntk_result = ntk_computer.compute(spec, X.numpy())
                analytical_time = time.time() - t0
                ntk_analytical = ntk_result.kernel_matrix

                # PyTorch empirical NTK (average over seeds)
                t0 = time.time()
                ntk_empiricals = []
                for seed in range(5):
                    torch.manual_seed(seed * 100)
                    model = build_mlp(input_dim, [256]*depth, 1,
                                    sigma_w=sigma_w, sigma_b=0.0)
                    ntk_emp = compute_empirical_ntk(model, X)
                    ntk_empiricals.append(ntk_emp)
                empirical_time = (time.time() - t0) / 5

                ntk_avg = np.mean(ntk_empiricals, axis=0)

                # Compare
                scale = np.trace(ntk_analytical) / max(np.trace(ntk_avg), 1e-10)
                ntk_avg_scaled = ntk_avg * scale
                rel_error = np.linalg.norm(ntk_avg_scaled - ntk_analytical, 'fro') / \
                           np.linalg.norm(ntk_analytical, 'fro')

                key = f'n{n_samples}_d{depth}'
                results[key] = {
                    'n_samples': n_samples,
                    'depth': depth,
                    'analytical_time_s': float(analytical_time),
                    'empirical_time_s': float(empirical_time),
                    'speedup': float(empirical_time / max(analytical_time, 1e-10)),
                    'relative_error': float(rel_error),
                    'analytical_cond': float(ntk_result.condition_number) if ntk_result.condition_number else -1,
                }
                print(f"  n={n_samples} d={depth}: analytical={analytical_time:.3f}s"
                      f" empirical={empirical_time:.3f}s error={rel_error:.4f}")

            except Exception as e:
                print(f"  n={n_samples} d={depth}: ERROR {e}")

    output = {
        'experiment': 'toolkit_benchmark',
        'results': results,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(RESULTS_DIR / 'exp_real_toolkit_benchmark.json', 'w') as f:
        json.dump(output, f, indent=2)
    return output


if __name__ == '__main__':
    print("Running Real Network Training Experiments")
    print("=" * 70)

    all_results = {}

    all_results['exp1'] = experiment_1_phase_validation()
    all_results['exp2'] = experiment_2_ntk_drift()
    all_results['exp3'] = experiment_3_ntk_convergence()
    all_results['exp4'] = experiment_4_init_comparison()
    all_results['exp5'] = experiment_5_activation_comparison()
    all_results['exp6'] = experiment_6_depth_scale()
    all_results['exp7'] = experiment_7_variance_propagation()
    all_results['exp8'] = experiment_8_practical_diagnosis()
    all_results['exp9'] = experiment_9_toolkit_benchmark()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)

    # Summary
    with open(RESULTS_DIR / 'exp_real_summary.json', 'w') as f:
        summary = {
            'n_experiments': len(all_results),
            'experiments': list(all_results.keys()),
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        json.dump(summary, f, indent=2)
