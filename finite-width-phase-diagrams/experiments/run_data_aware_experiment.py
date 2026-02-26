"""
Experiment: Dataset-aware phase classification validation.

Tests PhaseKit's kernel-task spectral alignment adjustment on real
tasks to validate that dataset properties shift the effective phase
boundary as predicted by Bordelon & Pehlevy (2020).

Tests:
  1. Well-aligned task (target in top kernel eigendirections)
  2. Poorly-aligned task (target orthogonal to top directions)
  3. Phase boundary shift quantification
  4. Comparison: data-aware vs architecture-only classification
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'data_aware')
os.makedirs(RESULTS_DIR, exist_ok=True)


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def generate_tasks(n_samples=200, input_dim=20, seed=42):
    """Generate tasks with varying alignment to the kernel.

    Returns list of (X, y, description, expected_alignment).
    """
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, input_dim)

    tasks = []

    # 1. Well-aligned: linear target (lives in top PC direction)
    w = np.zeros(input_dim)
    w[0] = 1.0
    y = X @ w + 0.1 * rng.randn(n_samples)
    tasks.append((X, y, 'linear_aligned', 'high'))

    # 2. Moderately aligned: low-rank quadratic
    y2 = X[:, 0]**2 + X[:, 1] * X[:, 2] + 0.1 * rng.randn(n_samples)
    tasks.append((X, y2, 'quadratic_moderate', 'medium'))

    # 3. Poorly aligned: target from high-frequency components
    y3 = np.sin(10 * X @ rng.randn(input_dim)) + 0.1 * rng.randn(n_samples)
    tasks.append((X, y3, 'highfreq_poor', 'low'))

    # 4. Random noise (no alignment)
    y4 = rng.randn(n_samples)
    tasks.append((X, y4, 'noise', 'none'))

    return tasks


def compute_spectral_alignment(kernel_eigenvalues, target_projections):
    """Compute kappa = sum(y_k^2 * lambda_k) / (||y||^2 * sum(lambda_k))."""
    lam = np.array(kernel_eigenvalues)
    proj = np.array(target_projections)
    numerator = np.sum(proj**2 * lam)
    denominator = np.sum(proj**2) * np.sum(lam)
    if denominator < 1e-30:
        return 0.0
    return float(numerator / denominator)


def run_experiment():
    """Run dataset-aware phase classification experiment."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available")
        return

    from data_aware_phase import DataAwarePhaseAnalyzer
    from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec

    tasks = generate_tasks()
    mf = MeanFieldAnalyzer()
    analyzer = DataAwarePhaseAnalyzer()

    results = {
        'experiment': 'data_aware_phase',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'configurations': [],
    }

    activations = ['relu', 'gelu', 'tanh']
    depths = [5, 10]
    widths = [64, 256]

    # Sigma_w values spanning the phase transition
    sigma_w_grid = {
        'relu': [0.5, 1.0, 1.414, 2.0, 2.5],
        'gelu': [0.5, 1.0, 1.5, 1.982, 2.5],
        'tanh': [0.5, 0.8, 1.010, 1.5, 2.0],
    }

    total_configs = 0
    data_aware_improvements = 0

    for act in activations:
        sw_star, _ = mf.find_edge_of_chaos(act)
        for depth in depths:
            for width in widths:
                for sw in sigma_w_grid[act]:
                    # Architecture-only analysis
                    spec = ArchitectureSpec(
                        depth=depth, width=width, activation=act,
                        sigma_w=sw, sigma_b=0.0
                    )
                    report = mf.analyze(spec)
                    arch_phase = report.phase
                    arch_chi = report.chi_1

                    for X, y, task_name, expected_align in tasks:
                        n = X.shape[0]
                        total_configs += 1

                        # Build a simple MLP for NTK computation
                        layers = []
                        for i in range(depth):
                            in_f = X.shape[1] if i == 0 else width
                            layers.append(nn.Linear(in_f, width))
                            if act == 'relu':
                                layers.append(nn.ReLU())
                            elif act == 'gelu':
                                layers.append(nn.GELU())
                            elif act == 'tanh':
                                layers.append(nn.Tanh())
                        layers.append(nn.Linear(width, 1))
                        model = nn.Sequential(*layers)

                        # Initialize with specified sigma_w
                        for m in model.modules():
                            if isinstance(m, nn.Linear):
                                nn.init.normal_(m.weight, 0, sw / np.sqrt(m.in_features))
                                if m.bias is not None:
                                    nn.init.zeros_(m.bias)

                        # Compute approximate NTK via random features
                        torch.manual_seed(42)
                        X_t = torch.tensor(X, dtype=torch.float32)
                        y_t = torch.tensor(y, dtype=torch.float32)

                        # Simple NTK approximation: use gradient features
                        model.eval()
                        n_sub = min(n, 50)
                        X_sub = X_t[:n_sub]
                        y_sub = y_t[:n_sub]

                        grads = []
                        for i in range(n_sub):
                            model.zero_grad()
                            out = model(X_sub[i:i+1])
                            out.backward()
                            g = torch.cat([p.grad.flatten()
                                          for p in model.parameters()
                                          if p.grad is not None])
                            grads.append(g.detach())
                            model.zero_grad()

                        G = torch.stack(grads).numpy()
                        K = G @ G.T

                        # Eigendecompose
                        eigvals, eigvecs = np.linalg.eigh(K)
                        eigvals = eigvals[::-1]
                        eigvecs = eigvecs[:, ::-1]

                        # Project target onto eigenbasis
                        y_proj = eigvecs.T @ y_sub.numpy()

                        # Compute spectral alignment
                        kappa = compute_spectral_alignment(
                            np.maximum(eigvals, 0), y_proj
                        )

                        # Data-aware chi adjustment
                        alpha = 0.5
                        chi_eff = arch_chi * (1 + alpha * (1 - kappa) * np.log(width) / width)

                        # Data-aware phase
                        if chi_eff < 0.9:
                            data_phase = 'ordered'
                        elif chi_eff > 1.1:
                            data_phase = 'chaotic'
                        else:
                            data_phase = 'critical'

                        # Train for a few steps to get ground truth trainability
                        model.train()
                        opt = torch.optim.SGD(model.parameters(), lr=0.01)
                        initial_loss = None
                        final_loss = None
                        for step in range(50):
                            opt.zero_grad()
                            pred = model(X_t).squeeze()
                            loss = ((pred - y_t)**2).mean()
                            if step == 0:
                                initial_loss = float(loss.item())
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            opt.step()
                        final_loss = float(loss.item())

                        loss_ratio = final_loss / max(initial_loss, 1e-10)
                        trained = loss_ratio < 0.9

                        cfg = {
                            'activation': act,
                            'depth': depth,
                            'width': width,
                            'sigma_w': sw,
                            'task': task_name,
                            'expected_alignment': expected_align,
                            'kappa': float(kappa),
                            'arch_chi': float(arch_chi),
                            'chi_eff': float(chi_eff),
                            'arch_phase': arch_phase,
                            'data_phase': data_phase,
                            'initial_loss': initial_loss,
                            'final_loss': final_loss,
                            'loss_ratio': float(loss_ratio),
                            'trained': trained,
                        }
                        results['configurations'].append(cfg)

    # Summary statistics
    configs = results['configurations']
    kappas_by_align = {}
    for c in configs:
        ea = c['expected_alignment']
        kappas_by_align.setdefault(ea, []).append(c['kappa'])

    results['summary'] = {
        'total_configs': len(configs),
        'mean_kappa_by_alignment': {
            k: float(np.mean(v)) for k, v in kappas_by_align.items()
        },
        'arch_phase_accuracy': None,  # filled below
        'data_phase_accuracy': None,
    }

    # Accuracy: does phase prediction match training outcome?
    arch_correct = 0
    data_correct = 0
    for c in configs:
        trained = c['trained']
        # "ordered" or "critical" should predict trainable for moderate sw
        arch_trainable = c['arch_phase'] in ('critical', 'ordered')
        data_trainable = c['data_phase'] in ('critical', 'ordered')

        if arch_trainable == trained:
            arch_correct += 1
        if data_trainable == trained:
            data_correct += 1

    n = len(configs)
    results['summary']['arch_phase_accuracy'] = float(arch_correct / n) if n > 0 else 0
    results['summary']['data_phase_accuracy'] = float(data_correct / n) if n > 0 else 0

    print(f"\n{'='*60}")
    print(f"Dataset-Aware Phase Classification Results")
    print(f"{'='*60}")
    print(f"Total configs: {len(configs)}")
    print(f"Mean kappa by alignment: {results['summary']['mean_kappa_by_alignment']}")
    print(f"Arch-only trainability accuracy: {results['summary']['arch_phase_accuracy']:.1%}")
    print(f"Data-aware trainability accuracy: {results['summary']['data_phase_accuracy']:.1%}")

    out_file = os.path.join(RESULTS_DIR, 'data_aware_results.json')
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {out_file}")


if __name__ == '__main__':
    run_experiment()
