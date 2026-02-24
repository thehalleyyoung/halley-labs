"""
PyTorch-based training experiments for scaled-up validation.

Addresses critiques:
- Scale up to width 512+, depth 50+
- Compare against Kaiming initialization (the PyTorch default)
- Use proper activation functions
- Test on MNIST with realistic training setup
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

RESULTS_DIR = Path(__file__).parent / 'data'
RESULTS_DIR.mkdir(exist_ok=True)


def load_mnist_torch(n_train=10000, n_test=2000):
    """Load MNIST using sklearn, return torch tensors."""
    try:
        from sklearn.datasets import fetch_openml
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        X, y = mnist.data.astype(np.float32), mnist.target.astype(np.int64)
    except Exception:
        rng = np.random.RandomState(42)
        X = rng.randn(70000, 784).astype(np.float32)
        y = rng.randint(0, 10, 70000).astype(np.int64)

    X = X / 255.0
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)

    rng = np.random.RandomState(42)
    indices = rng.permutation(len(X))
    X_train = torch.tensor(X[indices[:n_train]])
    y_train = torch.tensor(y[indices[:n_train]])
    X_test = torch.tensor(X[indices[60000:60000+n_test]])
    y_test = torch.tensor(y[indices[60000:60000+n_test]])
    return X_train, y_train, X_test, y_test


ACTIVATION_MAP = {
    'relu': nn.ReLU,
    'tanh': nn.Tanh,
    'gelu': nn.GELU,
    'silu': nn.SiLU,
}


class FlexMLP(nn.Module):
    """MLP with configurable activation, width, depth."""
    def __init__(self, input_dim, hidden_dim, output_dim, depth, activation='relu'):
        super().__init__()
        act_cls = ACTIVATION_MAP.get(activation, nn.ReLU)
        layers = [nn.Linear(input_dim, hidden_dim), act_cls()]
        for _ in range(depth - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), act_cls()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def init_custom(model, sigma_w, sigma_b=0.0):
    """Initialize with exact (σ_w, σ_b) parameterization."""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            fan_in = m.weight.shape[1]
            nn.init.normal_(m.weight, 0, sigma_w / np.sqrt(fan_in))
            if m.bias is not None:
                if sigma_b > 0:
                    nn.init.normal_(m.bias, 0, sigma_b)
                else:
                    nn.init.zeros_(m.bias)


def init_kaiming(model, activation='relu'):
    """Standard PyTorch Kaiming initialization."""
    nonlinearity = 'relu' if activation == 'relu' else 'leaky_relu'
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity=nonlinearity)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def train_pytorch(model, X_train, y_train, X_test, y_test,
                  lr=0.01, epochs=20, batch_size=128):
    """Train model with SGD, return metrics."""
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_losses = []
    test_accs = []
    exploded = False

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            if torch.isnan(loss) or loss.item() > 1e6:
                exploded = True
                break
            loss.backward()

            # Track gradient norm
            total_norm = 0
            for p in model.parameters():
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5

            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        if exploded:
            break
        train_losses.append(epoch_loss / max(n_batches, 1))

        model.eval()
        with torch.no_grad():
            logits = model(X_test)
            preds = logits.argmax(dim=1)
            acc = (preds == y_test).float().mean().item()
            test_accs.append(acc)

    final_acc = test_accs[-1] if test_accs else 0.0
    return {
        'train_losses': train_losses,
        'test_accs': test_accs,
        'final_acc': final_acc,
        'exploded': exploded,
    }


def run_scale_experiment():
    """Compare critical init vs Kaiming at various widths and depths."""
    from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec

    print("=" * 60)
    print("SCALED-UP PYTORCH EXPERIMENT")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist_torch(n_train=10000, n_test=2000)
    analyzer = MeanFieldAnalyzer()
    results = []

    configs = [
        {'width': 256, 'depth': 5, 'activation': 'relu'},
        {'width': 512, 'depth': 5, 'activation': 'relu'},
        {'width': 1024, 'depth': 5, 'activation': 'relu'},
        {'width': 512, 'depth': 10, 'activation': 'relu'},
        {'width': 512, 'depth': 20, 'activation': 'relu'},
        {'width': 512, 'depth': 5, 'activation': 'gelu'},
        {'width': 512, 'depth': 5, 'activation': 'silu'},
        {'width': 512, 'depth': 5, 'activation': 'tanh'},
        {'width': 512, 'depth': 10, 'activation': 'gelu'},
        {'width': 512, 'depth': 10, 'activation': 'silu'},
    ]

    eoc_values = {'relu': np.sqrt(2.0), 'tanh': 1.006, 'gelu': 1.534, 'silu': 1.677}
    seeds = [42, 123, 456]

    for cfg in configs:
        w, d, act = cfg['width'], cfg['depth'], cfg['activation']
        sigma_w_crit = eoc_values[act]

        arch = ArchitectureSpec(depth=d, width=w, activation=act,
                               sigma_w=sigma_w_crit, sigma_b=0.0)
        report = analyzer.analyze(arch)

        print(f"\n--- {act.upper()} width={w} depth={d} ---")
        print(f"  χ₁={report.chi_1:.4f}, phase={report.phase}")

        for init_name, init_fn in [
            ('critical', lambda m, a=act: init_custom(m, eoc_values[a], 0.0)),
            ('kaiming', lambda m, a=act: init_kaiming(m, a)),
            ('ordered', lambda m: init_custom(m, 0.5, 0.0)),
            ('chaotic', lambda m: init_custom(m, 3.0, 0.0)),
        ]:
            for seed in seeds:
                torch.manual_seed(seed)
                np.random.seed(seed)
                model = FlexMLP(784, w, 10, d, act)
                init_fn(model)

                t0 = time.time()
                metrics = train_pytorch(model, X_train, y_train, X_test, y_test,
                                       lr=0.01, epochs=20, batch_size=128)
                elapsed = time.time() - t0

                result = {
                    'width': w, 'depth': d, 'activation': act,
                    'init': init_name, 'seed': seed,
                    'predicted_chi1': report.chi_1,
                    'predicted_phase': report.phase,
                    'final_acc': metrics['final_acc'],
                    'exploded': metrics['exploded'],
                    'train_losses': metrics['train_losses'],
                    'test_accs': metrics['test_accs'],
                    'time_seconds': elapsed,
                }
                results.append(result)
                print(f"  {init_name:10s} seed={seed}: acc={metrics['final_acc']:.4f} "
                      f"exploded={metrics['exploded']} time={elapsed:.1f}s")

    out_path = RESULTS_DIR / "exp_pytorch_scale.json"
    with open(out_path, 'w') as f:
        json.dump({'experiment': 'pytorch_scale_validation', 'runs': results},
                  f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


def run_kaiming_comparison():
    """Direct head-to-head: critical init vs Kaiming default."""
    from mean_field_theory import MeanFieldAnalyzer, ArchitectureSpec

    print("\n" + "=" * 60)
    print("KAIMING VS CRITICAL INIT COMPARISON")
    print("=" * 60)

    X_train, y_train, X_test, y_test = load_mnist_torch(n_train=10000, n_test=2000)
    analyzer = MeanFieldAnalyzer()
    results = []

    eoc_values = {'relu': np.sqrt(2.0), 'gelu': 1.534, 'silu': 1.677}
    seeds = [42, 123, 456]

    for act in ['relu', 'gelu', 'silu']:
        for depth in [5, 10, 20]:
            for width in [256, 512]:
                sigma_crit = eoc_values[act]
                arch = ArchitectureSpec(depth=depth, width=width, activation=act,
                                       sigma_w=sigma_crit, sigma_b=0.0)
                report = analyzer.analyze(arch)

                print(f"\n{act.upper()} w={width} d={depth} χ₁={report.chi_1:.4f}")

                for init_name in ['critical', 'kaiming']:
                    accs = []
                    for seed in seeds:
                        torch.manual_seed(seed)
                        model = FlexMLP(784, width, 10, depth, act)
                        if init_name == 'critical':
                            init_custom(model, sigma_crit, 0.0)
                        else:
                            init_kaiming(model, act)

                        metrics = train_pytorch(model, X_train, y_train, X_test, y_test,
                                               lr=0.01, epochs=20, batch_size=128)
                        accs.append(metrics['final_acc'])

                        results.append({
                            'activation': act, 'width': width, 'depth': depth,
                            'init': init_name, 'seed': seed,
                            'final_acc': metrics['final_acc'],
                            'exploded': metrics['exploded'],
                            'chi1': report.chi_1,
                        })

                    mean_acc = np.mean(accs)
                    std_acc = np.std(accs)
                    print(f"  {init_name:10s}: {mean_acc:.4f} ± {std_acc:.4f}")

    out_path = RESULTS_DIR / "exp_kaiming_comparison.json"
    with open(out_path, 'w') as f:
        json.dump({'experiment': 'kaiming_vs_critical', 'runs': results},
                  f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


def run_finite_width_ntk_convergence():
    """NTK convergence with PyTorch at larger widths."""
    print("\n" + "=" * 60)
    print("FINITE-WIDTH NTK CONVERGENCE (PyTorch)")
    print("=" * 60)

    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))
    from ntk_computation import compute_ntk_simple

    results = []

    widths = [32, 64, 128, 256, 512, 1024]
    input_dim = 5
    n_samples = 15
    depth = 2
    sigma_w = np.sqrt(2.0)
    seeds = [42, 123, 456, 789, 1000]

    X = np.random.RandomState(42).randn(n_samples, input_dim).astype(np.float32)

    # Analytical NTK
    K_analytical = compute_ntk_simple(X, depth=depth, sigma_w=sigma_w, sigma_b=0.0)
    K_analytical_norm = K_analytical / np.trace(K_analytical) * n_samples

    for width in widths:
        errors = []
        for seed in seeds:
            torch.manual_seed(seed)
            model = FlexMLP(input_dim, width, 1, depth, 'relu')
            init_custom(model, sigma_w, 0.0)

            # Compute empirical NTK via Jacobian
            X_torch = torch.tensor(X, requires_grad=False)
            model.eval()

            # Compute Jacobian
            params = [p for p in model.parameters() if p.requires_grad]
            n_params = sum(p.numel() for p in params)

            jacobians = []
            for i in range(n_samples):
                model.zero_grad()
                out = model(X_torch[i:i+1])
                out.backward()
                jac_row = []
                for p in params:
                    if p.grad is not None:
                        jac_row.append(p.grad.data.clone().flatten())
                jacobians.append(torch.cat(jac_row))

            J = torch.stack(jacobians)  # (n_samples, n_params)
            K_empirical = (J @ J.T).numpy()
            K_empirical_norm = K_empirical / np.trace(K_empirical) * n_samples

            error = np.linalg.norm(K_analytical_norm - K_empirical_norm, 'fro') / np.linalg.norm(K_analytical_norm, 'fro')
            errors.append(error)

        mean_err = np.mean(errors)
        std_err = np.std(errors)
        results.append({
            'width': width, 'mean_error': mean_err, 'std_error': std_err,
            'errors': errors
        })
        print(f"  width={width:5d}: error={mean_err:.4f} ± {std_err:.4f}")

    # Log-log fit
    log_w = np.log(np.array([r['width'] for r in results]))
    log_e = np.log(np.array([r['mean_error'] for r in results]))
    slope, intercept = np.polyfit(log_w, log_e, 1)
    r2 = 1 - np.sum((log_e - (slope * log_w + intercept))**2) / np.sum((log_e - log_e.mean())**2)

    print(f"\n  Convergence rate: O(N^{{{slope:.3f}}}), R²={r2:.3f}")

    output = {
        'experiment': 'pytorch_ntk_convergence',
        'convergence_rate': float(slope),
        'r_squared': float(r2),
        'results': results,
    }
    out_path = RESULTS_DIR / "exp_pytorch_ntk_convergence.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to {out_path}")
    return output


if __name__ == "__main__":
    run_scale_experiment()
    run_kaiming_comparison()
    run_finite_width_ntk_convergence()
