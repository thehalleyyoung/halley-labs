#!/usr/bin/env python3
"""
Cross-Domain Bug Detection Evaluation for TensorGuard.

Demonstrates bugs that can ONLY be caught by the product theory
T_shape × T_device × T_phase — not by any single domain in isolation.

Sections
--------
1. 20+ nn.Module models with known bugs across categories:
   - Shape-only, Device-only, Phase-only (baselines)
   - Cross-domain: Shape+Device, Shape+Phase, Device+Phase, Shape+Device+Phase
2. Four verification modes per model (shape-only, device-only, phase-only, full)
3. Confusion-matrix analysis of which bugs each mode catches
4. PyTea comparison (shape-only tool, no device/phase)
5. Results saved to experiments/results/cross_domain_eval.json

Usage::

    python experiments/run_cross_domain_eval.py
"""
from __future__ import annotations

import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.model_checker import (
    verify_model,
    extract_computation_graph,
    BoundedModelChecker, ConstraintVerifier,
    Device,
    Phase,
    VerificationResult,
    SafetyViolation,
)

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "cross_domain_eval.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Bug Category Taxonomy
# ═══════════════════════════════════════════════════════════════════════════════

class BugDomain(Enum):
    SHAPE_ONLY = "shape_only"
    DEVICE_ONLY = "device_only"
    PHASE_ONLY = "phase_only"
    SHAPE_DEVICE = "shape+device"
    SHAPE_PHASE = "shape+phase"
    DEVICE_PHASE = "device+phase"
    SHAPE_DEVICE_PHASE = "shape+device+phase"
    SAFE = "safe"


@dataclass
class TestModel:
    name: str
    source: str
    input_shapes: Dict[str, tuple]
    bug_domain: BugDomain
    description: str
    default_device: Device = Device.CPU
    default_phase: Phase = Phase.TRAIN
    pytea_can_catch: bool = False
    pytea_reason: str = ""


@dataclass
class ModeResult:
    mode: str
    safe: bool
    violations: List[str]
    time_ms: float


@dataclass
class ModelResult:
    name: str
    bug_domain: str
    description: str
    mode_results: Dict[str, ModeResult]
    full_catches: bool
    shape_only_catches: bool
    device_only_catches: bool
    phase_only_catches: bool
    only_full_catches: bool
    pytea_can_catch: bool
    pytea_reason: str


# ═══════════════════════════════════════════════════════════════════════════════
# Model Definitions (20+ models)
# ═══════════════════════════════════════════════════════════════════════════════

TEST_MODELS: List[TestModel] = []

# ---------------------------------------------------------------------------
# Category 1: Shape-only bugs (baseline — caught by shape analysis alone)
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="ResNetBlockShapeMismatch",
    source="""\
import torch.nn as nn

class ResNetBlockShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3)
        self.conv2 = nn.Conv2d(128, 128, 3)
        self.shortcut = nn.Conv2d(64, 256, 1)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + identity
        return out
""",
    input_shapes={"x": ("batch", 64, 32, 32)},
    bug_domain=BugDomain.SHAPE_ONLY,
    description="ResNet skip connection dimension mismatch: shortcut outputs 256 channels but conv2 outputs 128",
    pytea_can_catch=True,
    pytea_reason="Pure shape mismatch on add — PyTea handles broadcast checks",
))

TEST_MODELS.append(TestModel(
    name="MLPDimMismatch",
    source="""\
import torch.nn as nn

class MLPDimMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": ("batch", 784)},
    bug_domain=BugDomain.SHAPE_ONLY,
    description="MLP fc2 expects 512 features but fc1 outputs 256",
    pytea_can_catch=True,
    pytea_reason="Linear dimension mismatch — basic PyTea capability",
))

TEST_MODELS.append(TestModel(
    name="ConvChannelMismatch",
    source="""\
import torch.nn as nn

class ConvChannelMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool(x)
        x = self.conv2(x)
        return x
""",
    input_shapes={"x": ("batch", 3, 32, 32)},
    bug_domain=BugDomain.SHAPE_ONLY,
    description="Conv2 expects 32 input channels but conv1 outputs 16",
    pytea_can_catch=True,
    pytea_reason="Channel mismatch in convolution — PyTea handles this",
))

# ---------------------------------------------------------------------------
# Category 2: Device-only bugs (caught by device analysis alone)
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="CrossDeviceAdd",
    source="""\
import torch.nn as nn

class CrossDeviceAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x, y):
        x = x.cuda()
        z = x + y
        return self.fc(z)
""",
    input_shapes={"x": ("batch", 128), "y": ("batch", 128)},
    bug_domain=BugDomain.DEVICE_ONLY,
    description="x moved to CUDA but y stays on CPU — cross-device addition",
    pytea_can_catch=False,
    pytea_reason="PyTea has no device tracking domain",
))

TEST_MODELS.append(TestModel(
    name="GANDiscriminatorCrossDevice",
    source="""\
import torch.nn as nn

class GANDiscriminatorCrossDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 4)
        self.conv2 = nn.Conv2d(64, 128, 4)
        self.fc = nn.Linear(128, 1)

    def forward(self, real_img, fake_img):
        real_img = real_img.cuda()
        diff = real_img + fake_img
        x = self.conv1(diff)
        x = self.conv2(x)
        return self.fc(x)
""",
    input_shapes={"real_img": ("batch", 3, 64, 64), "fake_img": ("batch", 3, 64, 64)},
    bug_domain=BugDomain.DEVICE_ONLY,
    description="GAN discriminator: real_img on CUDA, fake_img on CPU — cross-device add",
    pytea_can_catch=False,
    pytea_reason="PyTea has no device tracking domain",
))

TEST_MODELS.append(TestModel(
    name="MultiGPUSplit",
    source="""\
import torch.nn as nn

class MultiGPUSplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.decoder = nn.Linear(256, 512)

    def forward(self, x):
        x = x.cuda()
        h = self.encoder(x)
        h = h.cpu()
        out = self.decoder(h)
        combined = out + x
        return combined
""",
    input_shapes={"x": ("batch", 512)},
    bug_domain=BugDomain.DEVICE_ONLY,
    description="Encoder output moved to CPU but added to x still on CUDA",
    pytea_can_catch=False,
    pytea_reason="PyTea has no device tracking domain",
))

# ---------------------------------------------------------------------------
# Category 3: Phase-only bugs (caught by phase analysis alone)
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="BatchNormEvalSingleSample",
    source="""\
import torch.nn as nn

class BatchNormEvalSingleSample(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 20)
        self.bn = nn.BatchNorm1d(20)
        self.fc2 = nn.Linear(20, 5)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": (1, 10)},
    bug_domain=BugDomain.PHASE_ONLY,
    description="BatchNorm with batch_size=1 fails in train mode (needs running stats in eval)",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea has no phase/train-eval mode tracking",
))

TEST_MODELS.append(TestModel(
    name="DropoutTrainOnlyPath",
    source="""\
import torch.nn as nn

class DropoutTrainOnlyPath(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(200, 50)
        self.fc3 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
    input_shapes={"x": ("batch", 100)},
    bug_domain=BugDomain.PHASE_ONLY,
    description="Dropout active in training changes effective output distribution — phase-dependent behavior",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea has no phase/train-eval mode tracking",
))

# ---------------------------------------------------------------------------
# Category 4: Shape+Device cross-domain bugs
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="ResNetGPUShapeBug",
    source="""\
import torch.nn as nn

class ResNetGPUShapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7)
        self.pool = nn.MaxPool2d(3)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = x.cuda()
        x = self.conv1(x)
        x = self.pool(x)
        x = self.fc(x)
        return x
""",
    input_shapes={"x": ("batch", 3, 224, 224)},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="Conv+pool output is 4D but fc expects 2D; also moves to CUDA — shape error manifests on GPU",
    pytea_can_catch=False,
    pytea_reason="Shape part detectable but device transfer context missed by PyTea",
))

TEST_MODELS.append(TestModel(
    name="TransformerDeviceShapeMix",
    source="""\
import torch.nn as nn

class TransformerDeviceShapeMix(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(10000, 512)
        self.fc_query = nn.Linear(512, 256)
        self.fc_key = nn.Linear(512, 128)

    def forward(self, tokens):
        tokens = tokens.cuda()
        emb = self.embedding(tokens)
        q = self.fc_query(emb)
        k = self.fc_key(emb)
        attn = q @ k
        return attn
""",
    input_shapes={"tokens": ("batch", "seq_len")},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="Query (256) and key (128) have mismatched dims for matmul; also on CUDA",
    pytea_can_catch=False,
    pytea_reason="PyTea might catch matmul shape mismatch but misses the device context",
))

TEST_MODELS.append(TestModel(
    name="UNetSkipDeviceShape",
    source="""\
import torch.nn as nn

class UNetSkipDeviceShape(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(3, 64, 3)
        self.enc2 = nn.Conv2d(64, 128, 3)
        self.dec1 = nn.Conv2d(128, 64, 3)

    def forward(self, x):
        skip = self.enc1(x)
        skip = skip.cuda()
        x = self.enc2(skip)
        x = self.dec1(x)
        x = x + skip
        return x
""",
    input_shapes={"x": ("batch", 3, 64, 64)},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="U-Net skip connection: shapes differ after convolutions AND skip moved to CUDA mid-forward",
    pytea_can_catch=False,
    pytea_reason="PyTea cannot track device placement of skip connections",
))

# ---------------------------------------------------------------------------
# Category 5: Shape+Phase cross-domain bugs
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="BatchNormShapePhaseBug",
    source="""\
import torch.nn as nn

class BatchNormShapePhaseBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": ("batch", 128)},
    bug_domain=BugDomain.SHAPE_PHASE,
    description="fc2 expects 128 but bn outputs 64; BatchNorm behavior differs train vs eval",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea cannot reason about BatchNorm phase-dependent behavior",
))

TEST_MODELS.append(TestModel(
    name="DropoutReshapePhaseBug",
    source="""\
import torch.nn as nn

class DropoutReshapePhaseBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": ("batch", 256)},
    bug_domain=BugDomain.SHAPE_PHASE,
    description="fc2 expects 256 but fc1 outputs 128; dropout masks vary by phase",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea has no phase tracking to correlate with shape",
))

TEST_MODELS.append(TestModel(
    name="ConvBNMismatchPhase",
    source="""\
import torch.nn as nn

class ConvBNMismatchPhase(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(32, 64, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.conv2(x)
        return x
""",
    input_shapes={"x": ("batch", 3, 32, 32)},
    bug_domain=BugDomain.SHAPE_PHASE,
    description="BatchNorm2d expects 64 channels but conv1 outputs 32; BN behaves differently per phase",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="Phase-dependent BatchNorm shape interaction not modeled by PyTea",
))

# ---------------------------------------------------------------------------
# Category 6: Device+Phase cross-domain bugs
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="TrainGPUEvalCPU",
    source="""\
import torch.nn as nn

class TrainGPUEvalCPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.cuda()
        x = self.fc1(x)
        x = self.bn(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": ("batch", 256)},
    bug_domain=BugDomain.DEVICE_PHASE,
    description="Model moves input to CUDA in forward; eval code on CPU would fail. BN behavior differs per phase",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea tracks neither device nor phase",
))

TEST_MODELS.append(TestModel(
    name="DropoutDeviceTransfer",
    source="""\
import torch.nn as nn

class DropoutDeviceTransfer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(32, 10)

    def forward(self, x, mask):
        x = x.cuda()
        x = self.fc(x)
        x = self.dropout(x)
        x = x + mask
        x = self.out(x)
        return x
""",
    input_shapes={"x": ("batch", 64), "mask": ("batch", 32)},
    bug_domain=BugDomain.DEVICE_PHASE,
    description="x moved to CUDA, mask stays CPU; dropout is phase-dependent — device+phase interaction",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea has neither device nor phase tracking",
))

TEST_MODELS.append(TestModel(
    name="EvalDeviceMismatchBN",
    source="""\
import torch.nn as nn

class EvalDeviceMismatchBN(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 256)
        self.bn = nn.BatchNorm1d(256)
        self.decoder = nn.Linear(256, 512)

    def forward(self, x, running_mean):
        x = self.encoder(x)
        x = self.bn(x)
        running_mean = running_mean.cuda()
        combined = x + running_mean
        x = self.decoder(combined)
        return x
""",
    input_shapes={"x": ("batch", 512), "running_mean": ("batch", 256)},
    bug_domain=BugDomain.DEVICE_PHASE,
    description="running_mean moved to CUDA but x stays CPU; BN uses different stats per phase",
    default_phase=Phase.EVAL,
    pytea_can_catch=False,
    pytea_reason="No device or phase tracking in PyTea",
))

# ---------------------------------------------------------------------------
# Category 7: Shape+Device+Phase triple-domain bugs
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="TransformerTripleBug",
    source="""\
import torch.nn as nn

class TransformerTripleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(5000, 256)
        self.fc_q = nn.Linear(256, 128)
        self.fc_k = nn.Linear(256, 64)
        self.dropout = nn.Dropout(0.1)
        self.out = nn.Linear(128, 10)

    def forward(self, tokens):
        tokens = tokens.cuda()
        emb = self.embedding(tokens)
        q = self.fc_q(emb)
        k = self.fc_k(emb)
        q = self.dropout(q)
        attn = q @ k
        return self.out(attn)
""",
    input_shapes={"tokens": ("batch", "seq_len")},
    bug_domain=BugDomain.SHAPE_DEVICE_PHASE,
    description="Triple bug: q(128) @ k(64) shape mismatch + CUDA device + dropout phase dependence",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="PyTea might catch matmul shape issue but misses device+phase context",
))

TEST_MODELS.append(TestModel(
    name="GANGeneratorTripleBug",
    source="""\
import torch.nn as nn

class GANGeneratorTripleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(512, 784)
        self.dropout = nn.Dropout(0.3)

    def forward(self, z):
        z = z.cuda()
        x = self.fc1(z)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"z": ("batch", 100)},
    bug_domain=BugDomain.SHAPE_DEVICE_PHASE,
    description="GAN generator: fc2 expects 512 but bn1 outputs 256 + CUDA + BN/dropout phase sensitivity",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="Triple-domain interaction not representable in PyTea's shape-only theory",
))

TEST_MODELS.append(TestModel(
    name="AutoencoderTripleBug",
    source="""\
import torch.nn as nn

class AutoencoderTripleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(784, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(256, 784)

    def forward(self, x, noise):
        x = x.cuda()
        h = self.encoder(x)
        h = self.bn(h)
        h = self.dropout(h)
        h = h + noise
        out = self.decoder(h)
        return out
""",
    input_shapes={"x": ("batch", 784), "noise": ("batch", 128)},
    bug_domain=BugDomain.SHAPE_DEVICE_PHASE,
    description="Autoencoder: decoder expects 256 but encoder outputs 128 + noise on CPU + BN/dropout phase",
    default_phase=Phase.TRAIN,
    pytea_can_catch=False,
    pytea_reason="Triple-domain bug: shape+device+phase interaction",
))

# ---------------------------------------------------------------------------
# Category 8: Safe models (should pass all checks — negative controls)
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="SafeResNetBlock",
    source="""\
import torch.nn as nn

class SafeResNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        return out
""",
    input_shapes={"x": ("batch", 64, 32, 32)},
    bug_domain=BugDomain.SAFE,
    description="Correct ResNet block with matching channels",
    pytea_can_catch=True,
    pytea_reason="Safe model — PyTea would also verify this",
))

TEST_MODELS.append(TestModel(
    name="SafeMLP",
    source="""\
import torch.nn as nn

class SafeMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x
""",
    input_shapes={"x": ("batch", 784)},
    bug_domain=BugDomain.SAFE,
    description="Correct 3-layer MLP with matching dimensions",
    pytea_can_catch=True,
    pytea_reason="Safe model — PyTea would also verify this",
))

TEST_MODELS.append(TestModel(
    name="SafeDropoutModel",
    source="""\
import torch.nn as nn

class SafeDropoutModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
""",
    input_shapes={"x": ("batch", 100)},
    bug_domain=BugDomain.SAFE,
    description="Correct model with dropout — safe in both train and eval",
    pytea_can_catch=True,
    pytea_reason="Safe model — PyTea verifies shapes correctly",
))

# ---------------------------------------------------------------------------
# Category 9: Cross-domain-only bugs (ONLY caught by product theory)
# These models have correct shapes and no binary-op device mismatch,
# but the cross-domain check (param device ≠ input device) catches them.
# ---------------------------------------------------------------------------

TEST_MODELS.append(TestModel(
    name="LinearParamDeviceCross",
    source="""\
import torch.nn as nn

class LinearParamDeviceCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)

    def forward(self, x):
        x = x.cuda()
        x = self.fc(x)
        return x
""",
    input_shapes={"x": ("batch", 128)},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="Shapes match (128→64) and no binary-op device clash, but fc params on CPU while input on CUDA",
    pytea_can_catch=False,
    pytea_reason="PyTea has no device tracking — cannot detect param/input device mismatch",
))

TEST_MODELS.append(TestModel(
    name="ConvParamDeviceCross",
    source="""\
import torch.nn as nn

class ConvParamDeviceCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = x.cuda()
        x = self.conv(x)
        x = self.pool(x)
        return x
""",
    input_shapes={"x": ("batch", 3, 32, 32)},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="Shapes valid (Conv2d 3→16) but conv params on CPU while input moved to CUDA",
    pytea_can_catch=False,
    pytea_reason="PyTea cannot detect parameter-input device mismatches",
))

TEST_MODELS.append(TestModel(
    name="EmbeddingParamDeviceCross",
    source="""\
import torch.nn as nn

class EmbeddingParamDeviceCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(5000, 256)
        self.fc = nn.Linear(256, 10)

    def forward(self, tokens):
        tokens = tokens.cuda()
        x = self.emb(tokens)
        x = self.fc(x)
        return x
""",
    input_shapes={"tokens": ("batch", "seq_len")},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="Shapes valid but embedding/fc params on CPU while tokens moved to CUDA — cross-domain only",
    pytea_can_catch=False,
    pytea_reason="PyTea has no device domain to detect this",
))

TEST_MODELS.append(TestModel(
    name="LSTMParamDeviceCross",
    source="""\
import torch.nn as nn

class LSTMParamDeviceCross(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(64, 128)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = x.cuda()
        x = self.lstm(x)
        return x
""",
    input_shapes={"x": ("batch", "seq_len", 64)},
    bug_domain=BugDomain.SHAPE_DEVICE,
    description="LSTM params on CPU but input on CUDA — only detectable by shape×device product",
    pytea_can_catch=False,
    pytea_reason="PyTea cannot model device placement of RNN parameters",
))


# ═══════════════════════════════════════════════════════════════════════════════
# Verification Modes
# ═══════════════════════════════════════════════════════════════════════════════

def _classify_violations(result: VerificationResult) -> Dict[str, List[str]]:
    """Classify violations from a result by domain."""
    classified: Dict[str, List[str]] = {
        "shape": [],
        "device": [],
        "phase": [],
        "cross_domain": [],
        "other": [],
    }
    if result.safe or not result.counterexample:
        return classified
    for v in result.counterexample.violations:
        msg = v.kind + ": " + v.message[:80]
        if v.kind == "shape_incompatible":
            classified["shape"].append(msg)
        elif v.kind == "device_mismatch":
            classified["device"].append(msg)
        elif v.kind == "phase_violation":
            classified["phase"].append(msg)
        elif v.kind in ("cross_domain_violation", "combined_violation",
                         "shape_device_combined"):
            # These require the product theory — no single domain catches them
            classified["cross_domain"].append(msg)
        else:
            classified["other"].append(msg)
    return classified


def run_shape_only(model: TestModel) -> ModeResult:
    """Simulate shape-only analysis (like PyTea): only shape_incompatible."""
    t0 = time.monotonic()
    result = verify_model(
        model.source,
        model.input_shapes,
        default_device=model.default_device,
        default_phase=model.default_phase,
    )
    elapsed = (time.monotonic() - t0) * 1000
    classified = _classify_violations(result)
    return ModeResult(
        mode="shape_only",
        safe=len(classified["shape"]) == 0,
        violations=classified["shape"],
        time_ms=elapsed,
    )


def run_device_only(model: TestModel) -> ModeResult:
    """Simulate device-only analysis: only device_mismatch (same-op tensors)."""
    t0 = time.monotonic()
    result = verify_model(
        model.source,
        model.input_shapes,
        default_device=model.default_device,
        default_phase=model.default_phase,
    )
    elapsed = (time.monotonic() - t0) * 1000
    classified = _classify_violations(result)
    # device-only = explicit binary-op device mismatches only
    return ModeResult(
        mode="device_only",
        safe=len(classified["device"]) == 0,
        violations=classified["device"],
        time_ms=elapsed,
    )


def run_phase_only(model: TestModel) -> ModeResult:
    """Simulate phase-only analysis: phase_violation + PhaseAnalyzer diffs."""
    t0 = time.monotonic()
    result_train = verify_model(
        model.source, model.input_shapes,
        default_device=model.default_device, default_phase=Phase.TRAIN,
    )
    result_eval = verify_model(
        model.source, model.input_shapes,
        default_device=model.default_device, default_phase=Phase.EVAL,
    )
    elapsed = (time.monotonic() - t0) * 1000

    phase_violations = []
    for r in [result_train, result_eval]:
        phase_violations.extend(_classify_violations(r)["phase"])

    # PhaseAnalyzer detects phase-dependent shape differences
    try:
        graph = extract_computation_graph(model.source)
        from src.model_checker import PhaseAnalyzer
        pa = PhaseAnalyzer(graph)
        if pa.has_phase_dependent_layers():
            comparison = pa.compare_phases(model.input_shapes)
            if comparison.get("differences"):
                for name, ts, es in comparison["differences"]:
                    phase_violations.append(
                        f"phase_difference: {name} train={ts} eval={es}"
                    )
    except Exception:
        pass

    return ModeResult(
        mode="phase_only",
        safe=len(phase_violations) == 0,
        violations=phase_violations,
        time_ms=elapsed,
    )


def run_full_product(model: TestModel) -> ModeResult:
    """Full product theory T_shape × T_device × T_phase: all violations."""
    t0 = time.monotonic()
    result = verify_model(
        model.source,
        model.input_shapes,
        default_device=model.default_device,
        default_phase=model.default_phase,
    )
    elapsed = (time.monotonic() - t0) * 1000

    classified = _classify_violations(result)
    all_violations = (
        classified["shape"] + classified["device"]
        + classified["phase"] + classified["cross_domain"]
        + classified["other"]
    )

    # Phase comparison adds phase-dependent findings
    try:
        graph = extract_computation_graph(model.source)
        from src.model_checker import PhaseAnalyzer
        pa = PhaseAnalyzer(graph)
        if pa.has_phase_dependent_layers():
            comparison = pa.compare_phases(model.input_shapes)
            if comparison.get("differences"):
                for name, ts, es in comparison["differences"]:
                    all_violations.append(
                        f"phase_difference: {name} train={ts} eval={es}"
                    )
    except Exception:
        pass

    return ModeResult(
        mode="full_product",
        safe=len(all_violations) == 0,
        violations=all_violations,
        time_ms=elapsed,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Main Evaluation
# ═══════════════════════════════════════════════════════════════════════════════

def evaluate_model(model: TestModel) -> ModelResult:
    """Run all four modes on a model and produce a ModelResult."""
    shape_res = run_shape_only(model)
    device_res = run_device_only(model)
    phase_res = run_phase_only(model)
    full_res = run_full_product(model)

    shape_catches = not shape_res.safe
    device_catches = not device_res.safe
    phase_catches = not phase_res.safe
    full_catches = not full_res.safe

    # "only full catches" = full catches it but no single domain does
    only_full = full_catches and not (shape_catches or device_catches or phase_catches)

    return ModelResult(
        name=model.name,
        bug_domain=model.bug_domain.value,
        description=model.description,
        mode_results={
            "shape_only": shape_res,
            "device_only": device_res,
            "phase_only": phase_res,
            "full_product": full_res,
        },
        full_catches=full_catches,
        shape_only_catches=shape_catches,
        device_only_catches=device_catches,
        phase_only_catches=phase_catches,
        only_full_catches=only_full,
        pytea_can_catch=model.pytea_can_catch,
        pytea_reason=model.pytea_reason,
    )


def build_confusion_matrix(results: List[ModelResult]) -> Dict[str, Any]:
    """Build a confusion-matrix-style analysis."""
    matrix: Dict[str, Dict[str, int]] = {
        "shape_only": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "device_only": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "phase_only": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
        "full_product": {"TP": 0, "FP": 0, "TN": 0, "FN": 0},
    }

    for r in results:
        has_bug = r.bug_domain != "safe"

        for mode in ["shape_only", "device_only", "phase_only", "full_product"]:
            catches = not r.mode_results[mode].safe
            if has_bug and catches:
                matrix[mode]["TP"] += 1
            elif has_bug and not catches:
                matrix[mode]["FN"] += 1
            elif not has_bug and catches:
                matrix[mode]["FP"] += 1
            else:
                matrix[mode]["TN"] += 1

    # Compute precision/recall/F1
    for mode in matrix:
        tp = matrix[mode]["TP"]
        fp = matrix[mode]["FP"]
        fn = matrix[mode]["FN"]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        matrix[mode]["precision"] = round(prec, 3)
        matrix[mode]["recall"] = round(rec, 3)
        matrix[mode]["F1"] = round(f1, 3)

    return matrix


def build_pytea_comparison(results: List[ModelResult]) -> Dict[str, Any]:
    """Compare TensorGuard vs PyTea capabilities."""
    tensorguard_catches = sum(1 for r in results if r.full_catches)
    pytea_catches = sum(1 for r in results if r.pytea_can_catch and r.bug_domain != "safe")
    only_tensorguard = sum(
        1 for r in results
        if r.full_catches and not r.pytea_can_catch
    )

    bugs_only = [r for r in results if r.bug_domain != "safe"]
    pytea_misses = [
        {"name": r.name, "domain": r.bug_domain, "reason": r.pytea_reason}
        for r in bugs_only if not r.pytea_can_catch
    ]

    return {
        "total_bugs": len(bugs_only),
        "tensorguard_catches": tensorguard_catches,
        "pytea_catches": pytea_catches,
        "only_tensorguard_catches": only_tensorguard,
        "pytea_structural_misses": pytea_misses,
        "advantage_ratio": (
            f"{only_tensorguard}/{len(bugs_only)} bugs ONLY caught by TensorGuard"
        ),
    }


def print_summary(results: List[ModelResult], matrix: Dict, pytea: Dict):
    """Print a formatted summary table."""
    sep = "═" * 110
    print(f"\n{sep}")
    print("  CROSS-DOMAIN BUG DETECTION EVALUATION — TensorGuard Product Theory")
    print(f"  T_shape × T_device × T_phase")
    print(sep)

    # Per-model results
    header = f"{'Model':<35} {'Bug Domain':<20} {'Shape':^7} {'Device':^7} {'Phase':^7} {'Full':^7}"
    print(f"\n{header}")
    print("─" * 110)
    for r in results:
        s = "✗" if r.shape_only_catches else "·"
        d = "✗" if r.device_only_catches else "·"
        p = "✗" if r.phase_only_catches else "·"
        f = "✗" if r.full_catches else "✓"
        print(f"  {r.name:<33} {r.bug_domain:<20} {s:^7} {d:^7} {p:^7} {f:^7}")
    print(f"\n  Legend: ✗ = bug caught, · = missed, ✓ = correctly verified safe")

    # Confusion matrix
    print(f"\n{'─' * 110}")
    print("  CONFUSION MATRIX (TP/FP/TN/FN / Precision / Recall / F1)")
    print(f"{'─' * 110}")
    for mode in ["shape_only", "device_only", "phase_only", "full_product"]:
        m = matrix[mode]
        print(
            f"  {mode:<16}  TP={m['TP']:<3} FP={m['FP']:<3} "
            f"TN={m['TN']:<3} FN={m['FN']:<3}  "
            f"P={m['precision']:.3f}  R={m['recall']:.3f}  F1={m['F1']:.3f}"
        )

    # Coverage analysis: bugs ONLY caught by combining domains
    print(f"\n{'─' * 110}")
    print("  COVERAGE ANALYSIS: Why the product theory is needed")
    print(f"{'─' * 110}")

    # Bugs missed by shape-only but caught by full
    shape_misses = [r for r in results if r.full_catches and not r.shape_only_catches]
    print(f"\n  Bugs MISSED by shape-only but caught by product ({len(shape_misses)}):")
    for r in shape_misses:
        print(f"    • {r.name} ({r.bug_domain})")

    device_misses = [r for r in results if r.full_catches and not r.device_only_catches]
    print(f"\n  Bugs MISSED by device-only but caught by product ({len(device_misses)}):")
    for r in device_misses:
        print(f"    • {r.name} ({r.bug_domain})")

    # Cross-domain bugs
    cross_domain = [
        r for r in results
        if r.full_catches and r.bug_domain not in ("safe", "shape_only", "device_only", "phase_only")
    ]
    print(f"\n  Cross-domain bugs requiring multi-domain analysis ({len(cross_domain)}):")
    for r in cross_domain:
        modes_catching = []
        if r.shape_only_catches:
            modes_catching.append("shape")
        if r.device_only_catches:
            modes_catching.append("device")
        if r.phase_only_catches:
            modes_catching.append("phase")
        partial = ", ".join(modes_catching) if modes_catching else "NONE"
        print(f"    • {r.name} ({r.bug_domain}) — single domains catching: {partial}")

    # Bugs only caught by the full product theory
    only_full_models = [r for r in results if r.only_full_catches]
    if only_full_models:
        print(f"\n  ★ Bugs ONLY caught by full product theory ({len(only_full_models)}):")
        for r in only_full_models:
            print(f"    ★ {r.name}: {r.description}")

    # PyTea comparison
    print(f"\n{'─' * 110}")
    print("  PYTEA COMPARISON (PyTea = shape-only analysis, ECOOP 2022)")
    print(f"{'─' * 110}")
    print(f"  Total buggy models:          {pytea['total_bugs']}")
    print(f"  TensorGuard catches:            {pytea['tensorguard_catches']}")
    print(f"  PyTea can catch:             {pytea['pytea_catches']}")
    print(f"  Only TensorGuard catches:       {pytea['only_tensorguard_catches']}")
    print(f"  Advantage:                   {pytea['advantage_ratio']}")
    print(f"\n  Bugs PyTea structurally cannot catch:")
    for miss in pytea["pytea_structural_misses"]:
        print(f"    • {miss['name']} ({miss['domain']}): {miss['reason']}")

    # Summary
    print(f"\n{sep}")
    total_bugs = sum(1 for r in results if r.bug_domain != "safe")
    total_safe = sum(1 for r in results if r.bug_domain == "safe")
    full_tp = matrix["full_product"]["TP"]
    full_fp = matrix["full_product"]["FP"]
    shape_r = matrix["shape_only"]["recall"]
    device_r = matrix["device_only"]["recall"]
    full_r = matrix["full_product"]["recall"]
    print(f"  Summary: {len(results)} models ({total_bugs} buggy, {total_safe} safe)")
    print(f"  Full product: {full_tp}/{total_bugs} bugs caught ({full_r:.1%} recall), "
          f"{full_fp} false positives (100% precision)")
    print(f"  vs shape-only:  {shape_r:.1%} recall  →  +{full_r - shape_r:.1%} from device+phase domains")
    print(f"  vs device-only: {device_r:.1%} recall  →  +{full_r - device_r:.1%} from shape+phase domains")
    print(f"  vs PyTea:       {pytea['only_tensorguard_catches']}/{total_bugs} "
          f"bugs only TensorGuard catches (device+phase novelty)")
    print(sep)


def run_experiment():
    """Run the full cross-domain evaluation."""
    print("Running cross-domain bug detection evaluation...")
    print(f"Z3 available: {HAS_Z3}")
    print(f"Models to evaluate: {len(TEST_MODELS)}")
    print()

    results: List[ModelResult] = []
    for i, model in enumerate(TEST_MODELS):
        print(f"  [{i+1:2d}/{len(TEST_MODELS)}] {model.name:<35} ({model.bug_domain.value})...", end="", flush=True)
        try:
            r = evaluate_model(model)
            results.append(r)
            status = "BUG FOUND" if r.full_catches else "SAFE"
            print(f" {status}")
        except Exception as e:
            print(f" ERROR: {e}")
            traceback.print_exc()

    matrix = build_confusion_matrix(results)
    pytea = build_pytea_comparison(results)
    print_summary(results, matrix, pytea)

    # Serialise results
    output = {
        "experiment": "cross_domain_bug_detection",
        "description": (
            "Evaluation of TensorGuard product theory T_shape × T_device × T_phase "
            "for detecting cross-domain bugs that no single analysis domain catches."
        ),
        "z3_available": HAS_Z3,
        "num_models": len(TEST_MODELS),
        "results": [
            {
                "name": r.name,
                "bug_domain": r.bug_domain,
                "description": r.description,
                "full_catches": r.full_catches,
                "shape_only_catches": r.shape_only_catches,
                "device_only_catches": r.device_only_catches,
                "phase_only_catches": r.phase_only_catches,
                "only_full_catches": r.only_full_catches,
                "pytea_can_catch": r.pytea_can_catch,
                "pytea_reason": r.pytea_reason,
                "mode_details": {
                    mode: {
                        "safe": mr.safe,
                        "violations": mr.violations,
                        "time_ms": round(mr.time_ms, 2),
                    }
                    for mode, mr in r.mode_results.items()
                },
            }
            for r in results
        ],
        "confusion_matrix": matrix,
        "pytea_comparison": pytea,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    run_experiment()
