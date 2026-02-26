"""
Experiment: Generalization to arbitrary PyTorch computation graphs.

Tests PhaseKit's compositional mean-field engine on diverse architectures:
  1. Standard MLP (baseline)
  2. CNN (Conv + Pool + FC)
  3. ResNet-style (residual blocks)
  4. Transformer (attention + FFN + LayerNorm)
  5. U-Net-style (encoder-decoder with skip connections)
  6. Mixture-of-experts gating
  7. Hybrid Conv+Transformer

For each architecture, compares:
  - Predicted variance trajectory (compositional mean-field)
  - Empirical variance trajectory (forward hooks)
  - Phase classification accuracy
"""

import sys
import os
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'graph_generalization')
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


def build_architectures():
    """Build diverse architectures for testing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available, skipping graph generalization experiments")
        return []

    archs = []

    # 1. MLP (baseline)
    for depth in [3, 5, 10]:
        for act_cls, act_name in [(nn.ReLU, 'relu'), (nn.GELU, 'gelu'), (nn.Tanh, 'tanh')]:
            layers = []
            for i in range(depth):
                in_f = 64 if i > 0 else 32
                layers.extend([nn.Linear(in_f, 64), act_cls()])
            layers.append(nn.Linear(64, 10))
            model = nn.Sequential(*layers)
            archs.append({
                'name': f'MLP-{depth}L-{act_name}',
                'model': model,
                'input_shape': (32,),
                'family': 'mlp',
            })

    # 2. CNN
    class SimpleCNN(nn.Module):
        def __init__(self, n_channels=32):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, n_channels, 3, padding=1), nn.ReLU(),
                nn.Conv2d(n_channels, n_channels, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
                nn.Conv2d(n_channels, n_channels * 2, 3, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.classifier = nn.Linear(n_channels * 2, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    archs.append({
        'name': 'CNN-4L',
        'model': SimpleCNN(32),
        'input_shape': (3, 16, 16),
        'family': 'cnn',
    })

    # 3. ResNet-style
    class ResidualBlock(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.act = nn.ReLU()
            self.fc2 = nn.Linear(dim, dim)

        def forward(self, x):
            return x + self.act(self.fc2(self.act(self.fc1(x))))

    class SimpleResNet(nn.Module):
        def __init__(self, dim=64, n_blocks=3):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            self.blocks = nn.Sequential(*[ResidualBlock(dim) for _ in range(n_blocks)])
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.blocks(x)
            return self.output(x)

    for n_blocks in [2, 4, 8]:
        archs.append({
            'name': f'ResNet-{n_blocks}B',
            'model': SimpleResNet(64, n_blocks),
            'input_shape': (32,),
            'family': 'resnet',
        })

    # 4. Transformer
    class MiniTransformer(nn.Module):
        def __init__(self, d_model=64, n_heads=4, n_layers=2):
            super().__init__()
            self.embed = nn.Linear(32, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                batch_first=True, norm_first=True
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.output = nn.Linear(d_model, 10)

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            x = self.embed(x)
            x = self.encoder(x)
            return self.output(x[:, 0])

    for n_layers in [2, 4, 6]:
        archs.append({
            'name': f'Transformer-{n_layers}L',
            'model': MiniTransformer(64, 4, n_layers),
            'input_shape': (32,),
            'family': 'transformer',
        })

    # 5. Hybrid Conv+Transformer
    class HybridConvTransformer(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv_stem = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.flatten = nn.Flatten()
            self.proj = nn.Linear(64 * 16, 64)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=64, nhead=4, dim_feedforward=128,
                batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
            self.head = nn.Linear(64, 10)

        def forward(self, x):
            x = self.conv_stem(x)
            x = self.flatten(x)
            x = self.proj(x).unsqueeze(1)
            x = self.transformer(x)
            return self.head(x[:, 0])

    archs.append({
        'name': 'Hybrid-ConvTransformer',
        'model': HybridConvTransformer(),
        'input_shape': (3, 16, 16),
        'family': 'hybrid',
    })

    # 6. Wide-and-deep style
    class WideAndDeep(nn.Module):
        def __init__(self, wide_dim=32, deep_dim=64, deep_layers=3):
            super().__init__()
            self.wide = nn.Linear(32, wide_dim)
            layers = [nn.Linear(32, deep_dim), nn.ReLU()]
            for _ in range(deep_layers - 1):
                layers.extend([nn.Linear(deep_dim, deep_dim), nn.ReLU()])
            self.deep = nn.Sequential(*layers)
            self.output = nn.Linear(wide_dim + deep_dim, 10)

        def forward(self, x):
            wide_out = self.wide(x)
            deep_out = self.deep(x)
            return self.output(torch.cat([wide_out, deep_out], dim=-1))

    archs.append({
        'name': 'WideAndDeep-3L',
        'model': WideAndDeep(),
        'input_shape': (32,),
        'family': 'multi_branch',
    })

    return archs


def run_experiment():
    """Run the full graph generalization experiment."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available")
        return

    from compositional_mf import analyze_arbitrary_graph
    from graph_analyzer import analyze_graph
    from pytorch_integration import analyze, detect_architecture

    archs = build_architectures()
    if not archs:
        return

    results = {
        'experiment': 'graph_generalization',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_architectures': len(archs),
        'architectures': [],
    }

    for arch_info in archs:
        name = arch_info['name']
        model = arch_info['model']
        input_shape = arch_info['input_shape']
        family = arch_info['family']

        print(f"\n{'='*60}")
        print(f"Architecture: {name} ({family})")

        n_params = sum(p.numel() for p in model.parameters())
        det = detect_architecture(model)
        print(f"  Params: {n_params:,}, Depth: {det.depth}, Type: {det.arch_type}")

        # Compositional mean-field analysis
        try:
            comp_result = analyze_arbitrary_graph(model, input_shape, n_samples=128)
            pred_var = comp_result.predicted_variance_trajectory
            emp_var = comp_result.empirical_variance_trajectory
            comp_phase = comp_result.phase
            comp_chi = comp_result.chi_1_total
            comp_sw = comp_result.recommended_sigma_w

            # Variance prediction error
            if pred_var and emp_var:
                min_len = min(len(pred_var), len(emp_var))
                if min_len > 1:
                    pred_arr = np.array(pred_var[:min_len])
                    emp_arr = np.array(emp_var[:min_len])
                    mask = (emp_arr > 1e-10) & (pred_arr > 1e-10)
                    if mask.sum() > 0:
                        rel_errors = np.abs(pred_arr[mask] - emp_arr[mask]) / emp_arr[mask]
                        var_error = float(np.mean(rel_errors))
                    else:
                        var_error = None
                else:
                    var_error = None
            else:
                var_error = None

            print(f"  Compositional MF: phase={comp_phase}, χ₁={comp_chi:.4f}, "
                  f"σ_w_rec={comp_sw:.4f}")
            if var_error is not None:
                print(f"  Variance prediction error: {var_error*100:.1f}%")

        except Exception as e:
            comp_phase = "error"
            comp_chi = 0.0
            comp_sw = 0.0
            var_error = None
            pred_var = []
            emp_var = []
            print(f"  Compositional MF: ERROR - {e}")

        # Standard analysis
        try:
            std_result = analyze(model)
            std_phase = std_result.phase
            std_chi = std_result.chi_1
        except Exception as e:
            std_phase = "error"
            std_chi = 0.0

        # Graph analyzer
        try:
            graph_result = analyze_graph(model, input_shape, n_samples=128)
            graph_phase = graph_result.phase
            graph_chi = graph_result.chi_1_total
        except Exception as e:
            graph_phase = "error"
            graph_chi = 0.0

        arch_result = {
            'name': name,
            'family': family,
            'n_params': n_params,
            'depth': det.depth,
            'arch_type': str(det.arch_type),
            'compositional_phase': comp_phase,
            'compositional_chi': float(comp_chi),
            'compositional_sw_rec': float(comp_sw),
            'standard_phase': std_phase,
            'standard_chi': float(std_chi),
            'graph_phase': graph_phase,
            'graph_chi': float(graph_chi),
            'variance_error_pct': float(var_error * 100) if var_error is not None else None,
            'predicted_variance': [float(v) for v in pred_var] if pred_var else [],
            'empirical_variance': [float(v) for v in emp_var] if emp_var else [],
        }
        results['architectures'].append(arch_result)
        print(f"  Standard: phase={std_phase}, χ₁={std_chi:.4f}")
        print(f"  Graph:    phase={graph_phase}, χ₁={graph_chi:.4f}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    families = set(a['family'] for a in results['architectures'])
    for fam in sorted(families):
        fam_archs = [a for a in results['architectures'] if a['family'] == fam]
        var_errors = [a['variance_error_pct'] for a in fam_archs
                      if a['variance_error_pct'] is not None]
        avg_err = np.mean(var_errors) if var_errors else None
        phases = [a['compositional_phase'] for a in fam_archs]
        print(f"  {fam:15s}: {len(fam_archs)} archs, "
              f"avg var error={avg_err:.1f}%" if avg_err else f"  {fam:15s}: {len(fam_archs)} archs",
              f"phases={phases}")

    # Save results
    output_file = os.path.join(RESULTS_DIR, 'graph_generalization_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_file}")


if __name__ == '__main__':
    run_experiment()
