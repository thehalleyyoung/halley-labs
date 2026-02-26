"""
Universal graph experiment: Test DAG-based variance propagation on 50+ architectures.

Tests PhaseKit's DAG propagator on diverse real-world PyTorch architectures:
  - MLPs (3-20 layers, 4 activations)
  - CNNs (custom, VGG-style)
  - ResNets (custom, torchvision ResNet18/34/50)
  - DenseNet-style
  - Transformers (custom, Pre-LN, Post-LN)
  - UNet-style
  - MobileNet-style (depthwise separable)
  - Squeeze-and-Excitation (SE)
  - Wide-and-Deep / multi-branch
  - Inception-style (parallel branches with concat)
  - Feature Pyramid Network (FPN)
  - Gated architectures (GLU, gated MLP)
  - Mixture-of-Experts (MoE router)
  - Vision Transformer (ViT) blocks
  - Funnel / bottleneck architectures

For each architecture:
  1. DAG-based predicted variance trajectory
  2. Empirical variance trajectory (forward hooks)
  3. Phase classification (predicted vs empirical ground truth)
  4. Variance prediction error
"""

import sys
import os
import json
import time
import traceback
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'implementation', 'src'))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), 'results', 'universal_graph')
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


def determine_empirical_phase(model, input_shape, n_samples=256, seed=42):
    """Determine the empirical phase from variance trajectory."""
    import torch
    import torch.nn as nn

    torch.manual_seed(seed)
    x = torch.randn(n_samples, *input_shape)

    # Collect per-layer variances
    variances = []
    hooks = []

    def make_hook(idx):
        def hook_fn(module, inp, output):
            if isinstance(output, tuple):
                out = output[0]
            elif isinstance(output, torch.Tensor):
                out = output
            else:
                return
            variances.append(float(out.detach().float().var().item()))
        return hook_fn

    idx = 0
    for name, module in model.named_modules():
        children = list(module.children())
        if len(children) == 0 or isinstance(module, nn.MultiheadAttention):
            h = module.register_forward_hook(make_hook(idx))
            hooks.append(h)
            idx += 1

    model.eval()
    with torch.no_grad():
        try:
            model(x)
        except Exception:
            try:
                model(x, x)
            except Exception:
                for h in hooks:
                    h.remove()
                return "error", []

    for h in hooks:
        h.remove()

    if len(variances) < 2:
        return "critical", variances

    # Filter out layers with degenerate variance (at machine epsilon)
    valid_variances = [v for v in variances if v > 1e-8]
    if len(valid_variances) < 2:
        # If most variances are near zero, network is deeply ordered
        if variances[-1] < 1e-6 and variances[0] > 1e-3:
            return "ordered", variances
        return "critical", variances

    # Use end-to-end variance ratio as primary signal.
    # Per-layer ratios oscillate (e.g. Linear amplifies, ReLU halves), giving
    # misleading geometric means.  The overall first→last ratio is more robust.
    overall_ratio = valid_variances[-1] / valid_variances[0] if valid_variances[0] > 1e-10 else 1.0
    n_layers = len(valid_variances) - 1
    if n_layers > 0:
        per_layer_ratio = overall_ratio ** (1.0 / n_layers)
    else:
        per_layer_ratio = 1.0

    if per_layer_ratio < 0.92:
        return "ordered", variances
    elif per_layer_ratio > 1.15:
        return "chaotic", variances
    else:
        return "critical", variances


def build_all_architectures():
    """Build 50+ diverse architectures for testing."""
    try:
        import torch
        import torch.nn as nn
    except ImportError:
        print("PyTorch not available")
        return []

    archs = []

    # ================================================================
    # 1. MLPs (12 variants)
    # ================================================================
    for depth in [3, 5, 10, 20]:
        for act_cls, act_name in [(nn.ReLU, 'relu'), (nn.GELU, 'gelu'), (nn.Tanh, 'tanh')]:
            layers = []
            for i in range(depth):
                in_f = 64 if i > 0 else 32
                layers.extend([nn.Linear(in_f, 64), act_cls()])
            layers.append(nn.Linear(64, 10))
            archs.append({
                'name': f'MLP-{depth}L-{act_name}',
                'model': nn.Sequential(*layers),
                'input_shape': (32,),
                'family': 'mlp',
            })

    # ================================================================
    # 2. CNNs (4 variants)
    # ================================================================
    class SimpleCNN(nn.Module):
        def __init__(self, channels=32, n_conv=3, act=nn.ReLU):
            super().__init__()
            convs = [nn.Conv2d(3, channels, 3, padding=1), act()]
            for _ in range(n_conv - 1):
                convs.extend([nn.Conv2d(channels, channels, 3, padding=1), act()])
            convs.append(nn.AdaptiveAvgPool2d(1))
            self.features = nn.Sequential(*convs)
            self.classifier = nn.Linear(channels, 10)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            return self.classifier(x)

    for n_conv, act_cls, act_name in [
        (3, nn.ReLU, 'relu'), (5, nn.ReLU, 'relu'),
        (3, nn.GELU, 'gelu'), (3, nn.SiLU, 'silu')
    ]:
        archs.append({
            'name': f'CNN-{n_conv}L-{act_name}',
            'model': SimpleCNN(32, n_conv, act_cls),
            'input_shape': (3, 16, 16),
            'family': 'cnn',
        })

    # VGG-style (deeper CNN with pooling)
    class VGGBlock(nn.Module):
        def __init__(self, in_c, out_c, n_conv=2):
            super().__init__()
            layers = [nn.Conv2d(in_c, out_c, 3, padding=1), nn.ReLU()]
            for _ in range(n_conv - 1):
                layers.extend([nn.Conv2d(out_c, out_c, 3, padding=1), nn.ReLU()])
            layers.append(nn.MaxPool2d(2))
            self.block = nn.Sequential(*layers)

        def forward(self, x):
            return self.block(x)

    class MiniVGG(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                VGGBlock(3, 32, 2),
                VGGBlock(32, 64, 2),
            )
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.classifier(self.features(x))

    archs.append({
        'name': 'MiniVGG-4L',
        'model': MiniVGG(),
        'input_shape': (3, 16, 16),
        'family': 'cnn',
    })

    # ================================================================
    # 3. ResNets (6 variants)
    # ================================================================
    class ResidualBlock(nn.Module):
        def __init__(self, dim, act=nn.ReLU):
            super().__init__()
            self.fc1 = nn.Linear(dim, dim)
            self.act1 = act()
            self.fc2 = nn.Linear(dim, dim)
            self.act2 = act()

        def forward(self, x):
            return x + self.act2(self.fc2(self.act1(self.fc1(x))))

    class SimpleResNet(nn.Module):
        def __init__(self, dim=64, n_blocks=3, act=nn.ReLU):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            self.blocks = nn.Sequential(*[ResidualBlock(dim, act) for _ in range(n_blocks)])
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            x = self.input_proj(x)
            x = self.blocks(x)
            return self.output(x)

    for n_blocks, act_cls, act_name in [
        (2, nn.ReLU, 'relu'), (4, nn.ReLU, 'relu'), (8, nn.ReLU, 'relu'),
        (4, nn.GELU, 'gelu'), (4, nn.SiLU, 'silu'), (4, nn.Tanh, 'tanh'),
    ]:
        archs.append({
            'name': f'ResNet-{n_blocks}B-{act_name}',
            'model': SimpleResNet(64, n_blocks, act_cls),
            'input_shape': (32,),
            'family': 'resnet',
        })

    # Conv ResNet
    class ConvResBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(channels)
            self.relu = nn.ReLU()
            self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(channels)

        def forward(self, x):
            residual = x
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.bn2(self.conv2(out))
            return self.relu(out + residual)

    class ConvResNet(nn.Module):
        def __init__(self, channels=32, n_blocks=3):
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(3, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
            self.blocks = nn.Sequential(*[ConvResBlock(channels) for _ in range(n_blocks)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(channels, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    for n_blocks in [2, 4]:
        archs.append({
            'name': f'ConvResNet-{n_blocks}B',
            'model': ConvResNet(32, n_blocks),
            'input_shape': (3, 16, 16),
            'family': 'resnet',
        })

    # ================================================================
    # 4. DenseNet-style (2 variants)
    # ================================================================
    class DenseBlock(nn.Module):
        def __init__(self, in_features, growth_rate=16):
            super().__init__()
            self.fc = nn.Linear(in_features, growth_rate)
            self.act = nn.ReLU()

        def forward(self, x):
            return torch.cat([x, self.act(self.fc(x))], dim=-1)

    class MiniDenseNet(nn.Module):
        def __init__(self, input_dim=32, n_blocks=4, growth=16):
            super().__init__()
            blocks = []
            dim = input_dim
            for _ in range(n_blocks):
                blocks.append(DenseBlock(dim, growth))
                dim += growth
            self.blocks = nn.Sequential(*blocks)
            self.classifier = nn.Linear(dim, 10)

        def forward(self, x):
            x = self.blocks(x)
            return self.classifier(x)

    for n_blocks in [4, 8]:
        archs.append({
            'name': f'DenseNet-{n_blocks}B',
            'model': MiniDenseNet(32, n_blocks),
            'input_shape': (32,),
            'family': 'densenet',
        })

    # ================================================================
    # 5. Transformers (6 variants)
    # ================================================================
    class MiniTransformer(nn.Module):
        def __init__(self, d_model=64, n_heads=4, n_layers=2, norm_first=True):
            super().__init__()
            self.embed = nn.Linear(32, d_model)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
                batch_first=True, norm_first=norm_first
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
            'name': f'PreLN-Transformer-{n_layers}L',
            'model': MiniTransformer(64, 4, n_layers, norm_first=True),
            'input_shape': (32,),
            'family': 'transformer',
        })

    for n_layers in [2, 4]:
        archs.append({
            'name': f'PostLN-Transformer-{n_layers}L',
            'model': MiniTransformer(64, 4, n_layers, norm_first=False),
            'input_shape': (32,),
            'family': 'transformer',
        })

    # Custom attention block
    class SimpleAttentionBlock(nn.Module):
        def __init__(self, d=64, n_heads=4):
            super().__init__()
            self.ln1 = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, batch_first=True)
            self.ln2 = nn.LayerNorm(d)
            self.ffn = nn.Sequential(nn.Linear(d, d*4), nn.GELU(), nn.Linear(d*4, d))

        def forward(self, x):
            if x.dim() == 2:
                x = x.unsqueeze(1)
            h = self.ln1(x)
            h, _ = self.attn(h, h, h)
            x = x + h
            x = x + self.ffn(self.ln2(x))
            return x[:, 0]

    archs.append({
        'name': 'AttentionBlock-GELU',
        'model': nn.Sequential(nn.Linear(32, 64), SimpleAttentionBlock(64, 4), nn.Linear(64, 10)),
        'input_shape': (32,),
        'family': 'transformer',
    })

    # ================================================================
    # 6. UNet-style (2 variants)
    # ================================================================
    class MiniUNet(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Linear(32, dim), nn.ReLU())
            self.enc2 = nn.Sequential(nn.Linear(dim, dim // 2), nn.ReLU())
            self.bottleneck = nn.Sequential(nn.Linear(dim // 2, dim // 2), nn.ReLU())
            self.dec2 = nn.Sequential(nn.Linear(dim // 2, dim), nn.ReLU())
            # Skip connection: dec2 output + enc1 output
            self.dec1 = nn.Sequential(nn.Linear(dim * 2, dim), nn.ReLU())
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(e1)
            b = self.bottleneck(e2)
            d2 = self.dec2(b)
            d1 = self.dec1(torch.cat([d2, e1], dim=-1))  # skip connection
            return self.output(d1)

    archs.append({
        'name': 'MiniUNet',
        'model': MiniUNet(64),
        'input_shape': (32,),
        'family': 'unet',
    })

    class ConvUNet(nn.Module):
        def __init__(self, base_c=16):
            super().__init__()
            self.enc1 = nn.Sequential(nn.Conv2d(3, base_c, 3, padding=1), nn.ReLU())
            self.pool1 = nn.MaxPool2d(2)
            self.enc2 = nn.Sequential(nn.Conv2d(base_c, base_c*2, 3, padding=1), nn.ReLU())
            self.pool2 = nn.MaxPool2d(2)
            self.bottleneck = nn.Sequential(nn.Conv2d(base_c*2, base_c*2, 3, padding=1), nn.ReLU())
            self.up2 = nn.Upsample(scale_factor=2)
            self.dec2 = nn.Sequential(nn.Conv2d(base_c*4, base_c, 3, padding=1), nn.ReLU())
            self.up1 = nn.Upsample(scale_factor=2)
            self.dec1 = nn.Sequential(nn.Conv2d(base_c*2, base_c, 3, padding=1), nn.ReLU())
            self.out = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(base_c, 10))

        def forward(self, x):
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool1(e1))
            b = self.bottleneck(self.pool2(e2))
            d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
            d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
            return self.out(d1)

    archs.append({
        'name': 'ConvUNet',
        'model': ConvUNet(16),
        'input_shape': (3, 16, 16),
        'family': 'unet',
    })

    # ================================================================
    # 7. MobileNet-style (depthwise separable)
    # ================================================================
    class DepthwiseSeparable(nn.Module):
        def __init__(self, in_c, out_c):
            super().__init__()
            self.dw = nn.Conv2d(in_c, in_c, 3, padding=1, groups=in_c)
            self.bn1 = nn.BatchNorm2d(in_c)
            self.relu1 = nn.ReLU()
            self.pw = nn.Conv2d(in_c, out_c, 1)
            self.bn2 = nn.BatchNorm2d(out_c)
            self.relu2 = nn.ReLU()

        def forward(self, x):
            return self.relu2(self.bn2(self.pw(self.relu1(self.bn1(self.dw(x))))))

    class MiniMobileNet(nn.Module):
        def __init__(self, n_blocks=3):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            blocks = [DepthwiseSeparable(32, 32) for _ in range(n_blocks)]
            self.blocks = nn.Sequential(*blocks)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    for n_blocks in [3, 6]:
        archs.append({
            'name': f'MobileNet-{n_blocks}B',
            'model': MiniMobileNet(n_blocks),
            'input_shape': (3, 16, 16),
            'family': 'mobilenet',
        })

    # ================================================================
    # 8. Squeeze-and-Excitation (SE) blocks
    # ================================================================
    class SEBlock(nn.Module):
        def __init__(self, channels, reduction=4):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
            )
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // reduction),
                nn.ReLU(),
                nn.Linear(channels // reduction, channels),
                nn.Sigmoid(),
            )

        def forward(self, x):
            out = self.conv(x)
            w = self.se(out).unsqueeze(-1).unsqueeze(-1)
            return out * w + x  # SE gating + residual

    class MiniSENet(nn.Module):
        def __init__(self, n_blocks=3):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU())
            self.blocks = nn.Sequential(*[SEBlock(32) for _ in range(n_blocks)])
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.blocks(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    archs.append({
        'name': 'SENet-3B',
        'model': MiniSENet(3),
        'input_shape': (3, 16, 16),
        'family': 'senet',
    })

    # ================================================================
    # 9. Wide-and-Deep / multi-branch
    # ================================================================
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
            return self.output(torch.cat([self.wide(x), self.deep(x)], dim=-1))

    archs.append({
        'name': 'WideAndDeep-3L',
        'model': WideAndDeep(),
        'input_shape': (32,),
        'family': 'multi_branch',
    })

    # ================================================================
    # 10. Inception-style (parallel branches with concat)
    # ================================================================
    class InceptionBlock(nn.Module):
        def __init__(self, in_c, out_c_per_branch=16):
            super().__init__()
            self.branch1 = nn.Sequential(nn.Conv2d(in_c, out_c_per_branch, 1), nn.ReLU())
            self.branch3 = nn.Sequential(nn.Conv2d(in_c, out_c_per_branch, 3, padding=1), nn.ReLU())
            self.branch5 = nn.Sequential(nn.Conv2d(in_c, out_c_per_branch, 5, padding=2), nn.ReLU())
            self.branch_pool = nn.Sequential(
                nn.MaxPool2d(3, stride=1, padding=1),
                nn.Conv2d(in_c, out_c_per_branch, 1), nn.ReLU()
            )

        def forward(self, x):
            return torch.cat([self.branch1(x), self.branch3(x),
                              self.branch5(x), self.branch_pool(x)], dim=1)

    class MiniInception(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1), nn.ReLU())
            self.inc1 = InceptionBlock(32, 16)  # output: 64 channels
            self.inc2 = InceptionBlock(64, 16)  # output: 64 channels
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(64, 10)

        def forward(self, x):
            x = self.stem(x)
            x = self.inc1(x)
            x = self.inc2(x)
            x = self.pool(x).view(x.size(0), -1)
            return self.fc(x)

    archs.append({
        'name': 'MiniInception-2B',
        'model': MiniInception(),
        'input_shape': (3, 16, 16),
        'family': 'inception',
    })

    # ================================================================
    # 11. Gated architectures (GLU)
    # ================================================================
    class GatedMLP(nn.Module):
        def __init__(self, dim=64, depth=4):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            layers = []
            for _ in range(depth):
                layers.append(nn.Linear(dim, dim * 2))  # split into gate + value
            self.layers = nn.ModuleList(layers)
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            x = self.input_proj(x)
            for layer in self.layers:
                h = layer(x)
                gate, value = h.chunk(2, dim=-1)
                x = torch.sigmoid(gate) * value + x  # GLU + residual
            return self.output(x)

    archs.append({
        'name': 'GatedMLP-4L',
        'model': GatedMLP(64, 4),
        'input_shape': (32,),
        'family': 'gated',
    })

    class GatedMLP_SiLU(nn.Module):
        def __init__(self, dim=64, depth=4):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            self.gate_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
            self.up_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
            self.down_projs = nn.ModuleList([nn.Linear(dim, dim) for _ in range(depth)])
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            x = self.input_proj(x)
            for gate, up, down in zip(self.gate_projs, self.up_projs, self.down_projs):
                x = x + down(torch.nn.functional.silu(gate(x)) * up(x))
            return self.output(x)

    archs.append({
        'name': 'SwiGLU-MLP-4L',
        'model': GatedMLP_SiLU(64, 4),
        'input_shape': (32,),
        'family': 'gated',
    })

    # ================================================================
    # 12. Bottleneck / funnel architectures
    # ================================================================
    class BottleneckMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(32, 128), nn.ReLU(),
                nn.Linear(128, 32), nn.ReLU(),
                nn.Linear(32, 128), nn.ReLU(),
                nn.Linear(128, 32), nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.net(x)

    archs.append({
        'name': 'BottleneckMLP',
        'model': BottleneckMLP(),
        'input_shape': (32,),
        'family': 'bottleneck',
    })

    class PyramidMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(32, 256), nn.ReLU(),
                nn.Linear(256, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 32), nn.ReLU(),
                nn.Linear(32, 10),
            )

        def forward(self, x):
            return self.net(x)

    archs.append({
        'name': 'PyramidMLP',
        'model': PyramidMLP(),
        'input_shape': (32,),
        'family': 'bottleneck',
    })

    # ================================================================
    # 13. Highway Networks
    # ================================================================
    class HighwayLayer(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.fc = nn.Linear(dim, dim)
            self.gate = nn.Linear(dim, dim)
            self.act = nn.ReLU()

        def forward(self, x):
            h = self.act(self.fc(x))
            t = torch.sigmoid(self.gate(x))
            return t * h + (1 - t) * x

    class HighwayNet(nn.Module):
        def __init__(self, dim=64, n_layers=4):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            self.layers = nn.Sequential(*[HighwayLayer(dim) for _ in range(n_layers)])
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            return self.output(self.layers(self.input_proj(x)))

    archs.append({
        'name': 'HighwayNet-4L',
        'model': HighwayNet(64, 4),
        'input_shape': (32,),
        'family': 'highway',
    })

    # ================================================================
    # 14. PreNorm + PostNorm MLP (LayerNorm MLP)
    # ================================================================
    class PreNormMLP(nn.Module):
        def __init__(self, dim=64, depth=4):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            blocks = []
            for _ in range(depth):
                blocks.extend([nn.LayerNorm(dim), nn.Linear(dim, dim), nn.GELU()])
            self.blocks = nn.Sequential(*blocks)
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            return self.output(self.blocks(self.input_proj(x)))

    archs.append({
        'name': 'PreNormMLP-4L',
        'model': PreNormMLP(64, 4),
        'input_shape': (32,),
        'family': 'normed_mlp',
    })

    # ================================================================
    # 15. BatchNorm MLP
    # ================================================================
    class BNormMLP(nn.Module):
        def __init__(self, dim=64, depth=4):
            super().__init__()
            layers = [nn.Linear(32, dim)]
            for _ in range(depth):
                layers.extend([nn.BatchNorm1d(dim), nn.ReLU(), nn.Linear(dim, dim)])
            layers.append(nn.Linear(dim, 10))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    archs.append({
        'name': 'BNormMLP-4L',
        'model': BNormMLP(64, 4),
        'input_shape': (32,),
        'family': 'normed_mlp',
    })

    # ================================================================
    # 16. Dropout networks
    # ================================================================
    class DropoutMLP(nn.Module):
        def __init__(self, dim=64, depth=4, p=0.3):
            super().__init__()
            layers = [nn.Linear(32, dim), nn.ReLU()]
            for _ in range(depth - 1):
                layers.extend([nn.Dropout(p), nn.Linear(dim, dim), nn.ReLU()])
            layers.append(nn.Linear(dim, 10))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    archs.append({
        'name': 'DropoutMLP-4L-p0.3',
        'model': DropoutMLP(64, 4, 0.3),
        'input_shape': (32,),
        'family': 'dropout',
    })

    # ================================================================
    # 17. Mixed Conv+FC
    # ================================================================
    class ConvToFC(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d(4),
            )
            self.fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(64 * 16, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 10),
            )

        def forward(self, x):
            return self.fc(self.conv(x))

    archs.append({
        'name': 'ConvToFC',
        'model': ConvToFC(),
        'input_shape': (3, 16, 16),
        'family': 'hybrid',
    })

    # ================================================================
    # 18. Hybrid Conv+Transformer
    # ================================================================
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

    # ================================================================
    # 19. Activation-free networks (linear)
    # ================================================================
    class LinearNet(nn.Module):
        def __init__(self, depth=5):
            super().__init__()
            layers = [nn.Linear(32, 64)]
            for _ in range(depth - 1):
                layers.append(nn.Linear(64, 64))
            layers.append(nn.Linear(64, 10))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    archs.append({
        'name': 'LinearNet-5L',
        'model': LinearNet(5),
        'input_shape': (32,),
        'family': 'linear',
    })

    # ================================================================
    # 20. Very deep networks (stress test)
    # ================================================================
    class DeepResNet(nn.Module):
        def __init__(self, dim=64, n_blocks=16):
            super().__init__()
            self.input_proj = nn.Linear(32, dim)
            blocks = []
            for _ in range(n_blocks):
                blocks.extend([nn.Linear(dim, dim), nn.ReLU()])
            self.blocks = nn.Sequential(*blocks)
            self.output = nn.Linear(dim, 10)

        def forward(self, x):
            identity = self.input_proj(x)
            x = identity
            for i in range(0, len(self.blocks), 2):
                x = self.blocks[i+1](self.blocks[i](x)) + x  # residual
            return self.output(x)

    archs.append({
        'name': 'DeepResNet-16B',
        'model': DeepResNet(64, 16),
        'input_shape': (32,),
        'family': 'deep',
    })

    # ================================================================
    # 21. Multi-head multi-task
    # ================================================================
    class MultiTaskNet(nn.Module):
        def __init__(self, dim=64):
            super().__init__()
            self.shared = nn.Sequential(
                nn.Linear(32, dim), nn.ReLU(),
                nn.Linear(dim, dim), nn.ReLU(),
            )
            self.head1 = nn.Linear(dim, 10)
            self.head2 = nn.Linear(dim, 5)

        def forward(self, x):
            shared = self.shared(x)
            return self.head1(shared)  # just use first head for analysis

    archs.append({
        'name': 'MultiTaskNet',
        'model': MultiTaskNet(64),
        'input_shape': (32,),
        'family': 'multi_branch',
    })

    # ================================================================
    # 22. Try torchvision models if available
    # ================================================================
    try:
        import torchvision.models as models

        # ResNet18 (real torchvision)
        try:
            model = models.resnet18(weights=None)
            archs.append({
                'name': 'torchvision-ResNet18',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # MobileNetV2
        try:
            model = models.mobilenet_v2(weights=None)
            archs.append({
                'name': 'torchvision-MobileNetV2',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # DenseNet121
        try:
            model = models.densenet121(weights=None)
            archs.append({
                'name': 'torchvision-DenseNet121',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # VGG11
        try:
            model = models.vgg11(weights=None)
            archs.append({
                'name': 'torchvision-VGG11',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # EfficientNet-B0
        try:
            model = models.efficientnet_b0(weights=None)
            archs.append({
                'name': 'torchvision-EfficientNetB0',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # SqueezeNet
        try:
            model = models.squeezenet1_0(weights=None)
            archs.append({
                'name': 'torchvision-SqueezeNet',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # ShuffleNetV2
        try:
            model = models.shufflenet_v2_x0_5(weights=None)
            archs.append({
                'name': 'torchvision-ShuffleNetV2',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

        # ResNet50
        try:
            model = models.resnet50(weights=None)
            archs.append({
                'name': 'torchvision-ResNet50',
                'model': model,
                'input_shape': (3, 32, 32),
                'family': 'torchvision',
            })
        except Exception:
            pass

    except ImportError:
        pass

    return archs


def run_experiment():
    """Run the full universal graph experiment."""
    try:
        import torch
    except ImportError:
        print("PyTorch not available")
        return

    from dag_propagator import analyze_dag
    from compositional_mf import analyze_arbitrary_graph
    from graph_analyzer import analyze_graph

    archs = build_all_architectures()
    if not archs:
        return

    print(f"Testing {len(archs)} architectures")
    print("=" * 80)

    results = {
        'experiment': 'universal_graph',
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'n_architectures': len(archs),
        'architectures': [],
    }

    dag_correct = 0
    comp_correct = 0
    graph_correct = 0
    total_with_gt = 0

    for arch_info in archs:
        name = arch_info['name']
        model = arch_info['model']
        input_shape = arch_info['input_shape']
        family = arch_info['family']

        print(f"\n{'='*60}")
        print(f"Architecture: {name} ({family})")

        n_params = sum(p.numel() for p in model.parameters())

        # Ground truth: empirical phase
        gt_phase, gt_vars = determine_empirical_phase(model, input_shape)
        print(f"  Empirical phase: {gt_phase}")

        # 1. DAG-based analysis (NEW)
        dag_phase = "error"
        dag_chi = 0.0
        dag_var_error = None
        dag_n_nodes = 0
        dag_n_branches = 0
        dag_n_residual = 0
        try:
            dag_result = analyze_dag(model, input_shape, n_samples=128)
            dag_phase = dag_result.phase
            dag_chi = dag_result.chi_total
            dag_var_error = dag_result.variance_error_pct
            dag_n_nodes = dag_result.n_nodes
            dag_n_branches = dag_result.n_branches
            dag_n_residual = dag_result.n_residual
            print(f"  DAG:  phase={dag_phase}, χ₁={dag_chi:.4f}, "
                  f"nodes={dag_n_nodes}, branches={dag_n_branches}, "
                  f"residual={dag_n_residual}, var_err={dag_var_error:.1f}%")
        except Exception as e:
            print(f"  DAG:  ERROR - {e}")
            traceback.print_exc()

        # 2. Compositional MF (existing)
        comp_phase = "error"
        comp_chi = 0.0
        comp_var_error = None
        try:
            comp_result = analyze_arbitrary_graph(model, input_shape, n_samples=128)
            comp_phase = comp_result.phase
            comp_chi = comp_result.chi_1_total
            pred = comp_result.predicted_variance_trajectory
            emp = comp_result.empirical_variance_trajectory
            if pred and emp:
                min_len = min(len(pred), len(emp))
                if min_len > 1:
                    p = np.array(pred[:min_len])
                    e = np.array(emp[:min_len])
                    mask = (e > 1e-10) & (p > 1e-10)
                    if mask.sum() > 0:
                        comp_var_error = float(np.mean(np.abs(p[mask] - e[mask]) / e[mask]) * 100)
            print(f"  Comp: phase={comp_phase}, χ₁={comp_chi:.4f}"
                  + (f", var_err={comp_var_error:.1f}%" if comp_var_error else ""))
        except Exception as e:
            print(f"  Comp: ERROR - {e}")

        # 3. Graph analyzer (existing)
        graph_phase = "error"
        graph_chi = 0.0
        try:
            graph_result = analyze_graph(model, input_shape, n_samples=128)
            graph_phase = graph_result.phase
            graph_chi = graph_result.chi_1_total
            print(f"  Graph: phase={graph_phase}, χ₁={graph_chi:.4f}")
        except Exception as e:
            print(f"  Graph: ERROR - {e}")

        # Accuracy tracking
        if gt_phase != "error":
            total_with_gt += 1
            if dag_phase == gt_phase:
                dag_correct += 1
            if comp_phase == gt_phase:
                comp_correct += 1
            if graph_phase == gt_phase:
                graph_correct += 1

        arch_result = {
            'name': name,
            'family': family,
            'n_params': n_params,
            'empirical_phase': gt_phase,
            'dag_phase': dag_phase,
            'dag_chi': float(dag_chi),
            'dag_var_error_pct': dag_var_error,
            'dag_n_nodes': dag_n_nodes,
            'dag_n_branches': dag_n_branches,
            'dag_n_residual': dag_n_residual,
            'dag_correct': dag_phase == gt_phase if gt_phase != "error" else None,
            'comp_phase': comp_phase,
            'comp_chi': float(comp_chi),
            'comp_var_error_pct': comp_var_error,
            'comp_correct': comp_phase == gt_phase if gt_phase != "error" else None,
            'graph_phase': graph_phase,
            'graph_chi': float(graph_chi),
            'graph_correct': graph_phase == gt_phase if gt_phase != "error" else None,
        }
        results['architectures'].append(arch_result)

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Total architectures: {len(archs)}")
    print(f"With ground truth: {total_with_gt}")

    if total_with_gt > 0:
        print(f"\nPhase classification accuracy:")
        print(f"  DAG propagator:    {dag_correct}/{total_with_gt} = {dag_correct/total_with_gt*100:.1f}%")
        print(f"  Compositional MF:  {comp_correct}/{total_with_gt} = {comp_correct/total_with_gt*100:.1f}%")
        print(f"  Graph analyzer:    {graph_correct}/{total_with_gt} = {graph_correct/total_with_gt*100:.1f}%")

    # Per-family summary
    families = sorted(set(a['family'] for a in results['architectures']))
    print(f"\nPer-family accuracy (DAG):")
    for fam in families:
        fam_archs = [a for a in results['architectures'] if a['family'] == fam]
        correct = sum(1 for a in fam_archs if a.get('dag_correct') is True)
        total = sum(1 for a in fam_archs if a.get('dag_correct') is not None)
        var_errors = [a['dag_var_error_pct'] for a in fam_archs
                      if a['dag_var_error_pct'] is not None]
        avg_err = np.mean(var_errors) if var_errors else float('nan')
        print(f"  {fam:15s}: {correct}/{total} correct, "
              f"avg var error={avg_err:.1f}%")

    results['summary'] = {
        'total': len(archs),
        'with_ground_truth': total_with_gt,
        'dag_correct': dag_correct,
        'dag_accuracy': dag_correct / total_with_gt if total_with_gt > 0 else 0.0,
        'comp_correct': comp_correct,
        'comp_accuracy': comp_correct / total_with_gt if total_with_gt > 0 else 0.0,
        'graph_correct': graph_correct,
        'graph_accuracy': graph_correct / total_with_gt if total_with_gt > 0 else 0.0,
    }

    # Save
    output_file = os.path.join(RESULTS_DIR, 'universal_graph_results.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {output_file}")

    return results


if __name__ == '__main__':
    run_experiment()
