#!/usr/bin/env python3
"""
Model Checker Evaluation Suite for TensorGuard.

Comprehensive benchmarks comparing TensorGuard's constraint-based verifier +
CEGAR-based shape predicate discovery against PyTea (ECOOP 2022).

Sections
--------
1. PyTea Comparison Benchmarks (20+ tensor shape verification tasks)
2. Real-World nn.Module Architectures (MLP, CNN, Transformer, U-Net, GAN, …)
3. CEGAR Evaluation (predicate discovery & convergence)
4. Z3 Solver Statistics
5. Ablation Study (shape-only vs. model-checker vs. full pipeline vs. device)

Saves results to experiments/results/model_checker_eval.json.
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
    Device,
    Phase,
    VerificationResult,
)
from src.shape_cegar import (
    run_shape_cegar,
    verify_and_discover,
    CEGARStatus,
    ShapeCEGARResult,
)
from src.tensor_shapes import analyze_shapes

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

EXPERIMENTS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENTS_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = RESULTS_DIR / "model_checker_eval.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark infrastructure
# ═══════════════════════════════════════════════════════════════════════════════


class BenchmarkCategory(Enum):
    PYTEA_COMPARISON = "pytea_comparison"
    REAL_WORLD_ARCH = "real_world_architecture"
    CEGAR_EVAL = "cegar_evaluation"
    Z3_STATS = "z3_statistics"
    ABLATION = "ablation_study"


@dataclass
class PyTeaCapability:
    """Documents whether PyTea (ECOOP 2022) can handle a benchmark."""
    can_handle: bool
    reason: str
    pytea_category: str  # "basic_shape" | "reshape" | "matmul" | "unsupported"


@dataclass
class Benchmark:
    name: str
    source: str
    input_shapes: Dict[str, tuple]
    category: BenchmarkCategory
    expect_safe: bool
    description: str
    default_device: Device = Device.CPU
    default_phase: Phase = Phase.TRAIN
    pytea_info: Optional[PyTeaCapability] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class BenchmarkResult:
    name: str
    category: str
    description: str
    expect_safe: bool
    actual_safe: bool
    correct: bool
    verification_time_ms: float
    num_violations: int
    violation_kinds: List[str]
    num_steps: int
    num_layers: int
    certificate_k: int
    errors: List[str]
    pytea_info: Optional[Dict[str, Any]] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CEGARBenchmarkResult:
    name: str
    description: str
    is_safe: bool
    num_predicates_discovered: int
    predicates: List[str]
    iterations: int
    final_status: str
    contracts_inferred: List[str]
    total_time_ms: float
    iteration_details: List[Dict[str, Any]]
    real_bugs: List[str]


@dataclass
class Z3StatsResult:
    name: str
    num_z3_queries: int
    total_solve_time_ms: float
    per_query_times_ms: List[float]
    num_symbolic_dims: int
    num_constraints: int
    result_safe: bool


@dataclass
class AblationResult:
    name: str
    shape_only_safe: bool
    shape_only_time_ms: float
    shape_only_errors: int
    model_checker_safe: bool
    model_checker_time_ms: float
    model_checker_errors: int
    full_pipeline_safe: bool
    full_pipeline_time_ms: float
    full_pipeline_predicates: int
    full_pipeline_iterations: int
    device_on_safe: bool
    device_on_time_ms: float
    device_on_violations: int
    device_off_safe: bool
    device_off_time_ms: float
    device_off_violations: int


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1: PyTea Comparison Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════
#
# PyTea (ECOOP 2022) capabilities:
#   ✓ basic shape operations (zeros, ones, randn)
#   ✓ reshape / view
#   ✓ matmul / mm / bmm
#   ✓ element-wise operations with broadcasting
#   ✓ nn.Linear, nn.Conv2d shape propagation
#   ✗ device tracking (CPU/CUDA mismatch detection)
#   ✗ phase-dependent analysis (train/eval behaviour of dropout, batchnorm)
#   ✗ constraint-based verification of full architectures
#   ✗ CEGAR-based contract discovery
#   ✗ symbolic shape verification with arbitrary constraints

PYTEA_BENCHMARKS: List[Benchmark] = []


def _pt(name, source, input_shapes, expect_safe, description, *,
        pytea_can, pytea_reason, pytea_category, tags=None,
        default_device=Device.CPU, default_phase=Phase.TRAIN):
    PYTEA_BENCHMARKS.append(Benchmark(
        name=name, source=source, input_shapes=input_shapes,
        category=BenchmarkCategory.PYTEA_COMPARISON,
        expect_safe=expect_safe, description=description,
        default_device=default_device, default_phase=default_phase,
        pytea_info=PyTeaCapability(pytea_can, pytea_reason, pytea_category),
        tags=tags or [],
    ))


# --- 1.1  Basic shape ops (PyTea ✓, TensorGuard ✓) -------------------------

_pt("pt01_linear_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", 10)}, True,
    "Correct nn.Linear with matching input dim",
    pytea_can=True, pytea_reason="Basic Linear shape check", pytea_category="basic_shape")

_pt("pt02_linear_mismatch", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", 8)}, False,
    "nn.Linear dimension mismatch: in_features=10 but input has dim 8",
    pytea_can=True, pytea_reason="Basic Linear shape check", pytea_category="basic_shape")

_pt("pt03_two_linears_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""", {"x": ("batch", 784)}, True,
    "Two-layer MLP with correct chain",
    pytea_can=True, pytea_reason="Linear chain shape propagation", pytea_category="basic_shape")

_pt("pt04_two_linears_mismatch", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 128)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""", {"x": ("batch", 784)}, False,
    "Linear chain mismatch: fc1 outputs 128 but fc2 expects 64",
    pytea_can=True, pytea_reason="Linear chain shape propagation", pytea_category="basic_shape")

# --- 1.2  Reshape / view (PyTea ✓, TensorGuard ✓) ---------------------------

_pt("pt05_reshape_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(784, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
""", {"x": ("batch", 1, 28, 28)}, True,
    "Flatten via view(-1) then Linear(784,10)",
    pytea_can=True, pytea_reason="Reshape with -1 inference", pytea_category="reshape")

_pt("pt06_reshape_wrong_total", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(500, 10)
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.fc(x)
""", {"x": ("batch", 1, 28, 28)}, False,
    "Flatten produces 784 but Linear expects 500",
    pytea_can=True, pytea_reason="Reshape + Linear mismatch", pytea_category="reshape")

# --- 1.3  Matmul (PyTea ✓, TensorGuard ✓) -----------------------------------

_pt("pt07_matmul_correct", """
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(64, 32)
    def forward(self, x):
        return self.w(x)
""", {"x": ("batch", 64)}, True,
    "Simple matmul through Linear: (batch,64) @ (64,32)",
    pytea_can=True, pytea_reason="Matmul inner dim check", pytea_category="matmul")

_pt("pt08_matmul_inner_mismatch", """
import torch
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Linear(64, 32)
    def forward(self, x):
        return self.w(x)
""", {"x": ("batch", 48)}, False,
    "Matmul inner dim mismatch: input last dim 48 != 64",
    pytea_can=True, pytea_reason="Matmul inner dim check", pytea_category="matmul")

# --- 1.4  Conv2d (PyTea ✓, TensorGuard ✓) -----------------------------------

_pt("pt09_conv2d_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)
""", {"x": ("batch", 3, 32, 32)}, True,
    "Conv2d with correct input channels=3",
    pytea_can=True, pytea_reason="Conv2d channel check", pytea_category="basic_shape")

_pt("pt10_conv2d_channel_mismatch", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, padding=1)
    def forward(self, x):
        return self.conv(x)
""", {"x": ("batch", 1, 32, 32)}, False,
    "Conv2d channel mismatch: expects 3 channels but input has 1",
    pytea_can=True, pytea_reason="Conv2d channel check", pytea_category="basic_shape")

# --- 1.5  Broadcasting (PyTea ✓, TensorGuard ✓) -----------------------------

_pt("pt11_broadcast_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        h = self.fc(x)
        return h + x
""", {"x": ("batch", 10)}, True,
    "Residual addition: (batch,10) + (batch,10) broadcasts correctly",
    pytea_can=True, pytea_reason="Broadcast check", pytea_category="basic_shape")

# --- 1.6  Cross-device (PyTea ✗, TensorGuard ✓) -----------------------------

_pt("pt12_device_mismatch_add", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):
        h = self.fc(x)
        y = x.cuda()
        return h + y
""", {"x": ("batch", 10)}, False,
    "Device mismatch: h on CPU, y moved to CUDA, then added",
    pytea_can=False, pytea_reason="PyTea does not track device placement",
    pytea_category="unsupported", tags=["device"])

_pt("pt13_device_mismatch_matmul", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)
    def forward(self, x):
        h = self.fc1(x)
        h = h.cuda()
        return self.fc2(h)
""", {"x": ("batch", 10)}, False,
    "Device mismatch: h moved to CUDA but fc2 weights are on CPU",
    pytea_can=False, pytea_reason="No device tracking",
    pytea_category="unsupported", tags=["device"])

_pt("pt14_device_correct_both_cuda", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 5)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", 10)}, True,
    "Both input and model on same device (CPU)",
    pytea_can=False, pytea_reason="No device tracking (but would not report error)",
    pytea_category="unsupported", tags=["device"])

# --- 1.7  Phase-dependent analysis (PyTea ✗, TensorGuard ✓) -----------------

_pt("pt15_dropout_train_eval", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.drop = nn.Dropout(0.5)
    def forward(self, x):
        h = self.fc(x)
        h = self.drop(h)
        return h
""", {"x": ("batch", 10)}, True,
    "Dropout is shape-preserving in both train and eval phases",
    pytea_can=False, pytea_reason="No phase-dependent analysis",
    pytea_category="unsupported", tags=["phase"])

_pt("pt16_batchnorm_train_eval", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(10, 10)
        self.bn = nn.BatchNorm1d(10)
    def forward(self, x):
        h = self.fc(x)
        h = self.bn(h)
        return h
""", {"x": ("batch", 10)}, True,
    "BatchNorm1d tracks running stats differently in train vs eval",
    pytea_can=False, pytea_reason="No phase-dependent analysis",
    pytea_category="unsupported", tags=["phase"])

# --- 1.8  Full architecture verification (PyTea ✗, TensorGuard ✓) -----------

_pt("pt17_resnet_block", """
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        return self.relu(out)
""", {"x": ("batch", 64, 32, 32)}, True,
    "Full ResNet residual block with skip connection",
    pytea_can=False,
    pytea_reason="PyTea cannot verify full architectures with constraint verification",
    pytea_category="unsupported", tags=["architecture"])

_pt("pt18_transformer_attention", """
import torch.nn as nn
class SelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(512, 1536)
        self.proj = nn.Linear(512, 512)
    def forward(self, x):
        qkv = self.qkv(x)
        return self.proj(qkv)
""", {"x": ("batch", "seq_len", 512)}, False,
    "Attention: qkv outputs 1536 but proj expects 512 — dimension error",
    pytea_can=False,
    pytea_reason="No constraint-based verification of attention mechanisms",
    pytea_category="unsupported", tags=["architecture"])

# --- 1.9  Symbolic shape constraints (PyTea ✗, TensorGuard ✓) ---------------

_pt("pt19_symbolic_constraint_linear", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", "features")}, True,
    "Symbolic input dim 'features' — CEGAR discovers features==768",
    pytea_can=False,
    pytea_reason="No symbolic shape verification with arbitrary constraints",
    pytea_category="unsupported", tags=["symbolic", "cegar"])

_pt("pt20_symbolic_constraint_chain", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc3(h)
""", {"x": ("batch", "d_in")}, True,
    "3-layer chain with symbolic input — CEGAR discovers d_in==256",
    pytea_can=False,
    pytea_reason="No symbolic constraint discovery",
    pytea_category="unsupported", tags=["symbolic", "cegar"])

# --- 1.10  Additional coverage -------------------------------------------

_pt("pt21_conv_chain_correct", """
import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        return self.conv3(h)
""", {"x": ("batch", 3, 32, 32)}, True,
    "3-layer Conv2d chain with correct channel progression",
    pytea_can=True, pytea_reason="Conv2d chain check", pytea_category="basic_shape")

_pt("pt22_conv_chain_mismatch", """
import torch.nn as nn
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 64, 3, padding=1)
    def forward(self, x):
        h = self.conv1(x)
        return self.conv2(h)
""", {"x": ("batch", 3, 32, 32)}, False,
    "Conv chain mismatch: conv1 outputs 32 channels but conv2 expects 16",
    pytea_can=True, pytea_reason="Conv2d chain check", pytea_category="basic_shape")

_pt("pt23_embedding_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        h = self.embed(x)
        return self.fc(h)
""", {"x": ("batch", "seq_len")}, True,
    "Embedding(1000,128) -> Linear(128,10)",
    pytea_can=True, pytea_reason="Embedding shape propagation", pytea_category="basic_shape")

_pt("pt24_embedding_linear_mismatch", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 128)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.embed(x)
        return self.fc(h)
""", {"x": ("batch", "seq_len")}, False,
    "Embedding outputs 128 but Linear expects 64",
    pytea_can=True, pytea_reason="Embedding + Linear mismatch", pytea_category="basic_shape")

_pt("pt25_layernorm_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 256)
        self.ln = nn.LayerNorm(256)
    def forward(self, x):
        h = self.fc(x)
        return self.ln(h)
""", {"x": ("batch", 256)}, True,
    "LayerNorm(256) after Linear(256,256)",
    pytea_can=True, pytea_reason="LayerNorm shape check", pytea_category="basic_shape")

_pt("pt26_multi_device_transfer", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 10)
        self.fc2 = nn.Linear(10, 5)
    def forward(self, x):
        h = self.fc1(x)
        h = h.cuda()
        h = h.cpu()
        return self.fc2(h)
""", {"x": ("batch", 10)}, True,
    "Device round-trip: CPU -> CUDA -> CPU, then fc2 on CPU",
    pytea_can=False, pytea_reason="No device tracking",
    pytea_category="unsupported", tags=["device"])

_pt("pt27_sequential_correct", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.relu(h)
        return self.fc2(h)
""", {"x": ("batch", 100)}, True,
    "Linear -> ReLU -> Linear, all correct",
    pytea_can=True, pytea_reason="Basic sequential check", pytea_category="basic_shape")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2: Real-World nn.Module Architectures
# ═══════════════════════════════════════════════════════════════════════════════

ARCH_BENCHMARKS: List[Benchmark] = []


def _arch(name, source, input_shapes, expect_safe, description, *,
          tags=None, default_device=Device.CPU, default_phase=Phase.TRAIN):
    ARCH_BENCHMARKS.append(Benchmark(
        name=name, source=source, input_shapes=input_shapes,
        category=BenchmarkCategory.REAL_WORLD_ARCH,
        expect_safe=expect_safe, description=description,
        default_device=default_device, default_phase=default_phase,
        tags=tags or [],
    ))


# --- 2.1  Simple MLP -----------------------------------------------------

_arch("arch01_mlp_correct", """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.drop(h)
        h = self.relu(self.fc2(h))
        h = self.drop(h)
        return self.fc3(h)
""", {"x": ("batch", 784)}, True,
    "3-layer MLP with dropout, all dimensions correct",
    tags=["mlp"])

_arch("arch02_mlp_bug_wrong_dim", """
import torch.nn as nn
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.fc2(h)
        return self.fc3(h)
""", {"x": ("batch", 784)}, False,
    "MLP bug: fc1 outputs 512 but fc2 expects 256",
    tags=["mlp", "bug"])

# --- 2.2  CNN (mini-ResNet block) -----------------------------------------

_arch("arch03_cnn_resblock_correct", """
import torch.nn as nn
class MiniResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.res_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.res_bn1 = nn.BatchNorm2d(64)
        self.res_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.res_bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        residual = h
        h = self.relu(self.res_bn1(self.res_conv1(h)))
        h = self.res_bn2(self.res_conv2(h))
        h = h + residual
        return self.relu(h)
""", {"x": ("batch", 3, 224, 224)}, True,
    "Mini-ResNet: stem + one residual block, correct",
    tags=["cnn", "resnet"])

_arch("arch04_cnn_channel_bug", """
import torch.nn as nn
class BuggyResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    def forward(self, x):
        h = self.conv1(x)
        return self.conv2(h)
""", {"x": ("batch", 3, 32, 32)}, False,
    "CNN bug: conv1 outputs 64 channels but conv2 expects 32",
    tags=["cnn", "bug"])

# --- 2.3  Transformer encoder block --------------------------------------

_arch("arch05_transformer_encoder_correct", """
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_qkv = nn.Linear(512, 1536)
        self.attn_proj = nn.Linear(512, 512)
        self.ln1 = nn.LayerNorm(512)
        self.ffn1 = nn.Linear(512, 2048)
        self.ffn2 = nn.Linear(2048, 512)
        self.ln2 = nn.LayerNorm(512)
        self.drop = nn.Dropout(0.1)
    def forward(self, x):
        h = self.ln1(x)
        qkv = self.attn_qkv(h)
        h = self.attn_proj(h)
        h = self.drop(h)
        x = x + h
        h = self.ln2(x)
        h = self.ffn1(h)
        h = self.ffn2(h)
        h = self.drop(h)
        return x + h
""", {"x": ("batch", "seq_len", 512)}, True,
    "Transformer encoder block: self-attention + FFN with residuals",
    tags=["transformer"])

_arch("arch06_transformer_ffn_bug", """
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(512)
        self.ffn1 = nn.Linear(512, 2048)
        self.ffn2 = nn.Linear(1024, 512)
    def forward(self, x):
        h = self.ln1(x)
        h = self.ffn1(h)
        h = self.ffn2(h)
        return x + h
""", {"x": ("batch", "seq_len", 512)}, False,
    "Transformer bug: ffn1 outputs 2048 but ffn2 expects 1024",
    tags=["transformer", "bug"])

# --- 2.4  U-Net-style encoder-decoder ------------------------------------

_arch("arch07_unet_encoder_correct", """
import torch.nn as nn
class UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(64, 32, 3, padding=1)
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(e1))
        d2 = self.relu(self.dec2(e2))
        return self.dec1(d2)
""", {"x": ("batch", 1, 128, 128)}, True,
    "Simplified U-Net encoder-decoder, correct channel flow",
    tags=["unet"])

_arch("arch08_unet_decoder_bug", """
import torch.nn as nn
class UNetBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.dec2 = nn.Conv2d(128, 32, 3, padding=1)
        self.dec1 = nn.Conv2d(32, 1, 3, padding=1)
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d2 = self.dec2(e2)
        return self.dec1(d2)
""", {"x": ("batch", 1, 128, 128)}, False,
    "U-Net bug: dec2 expects 128 channels but enc2 only outputs 64",
    tags=["unet", "bug"])

# --- 2.5  GAN (generator + discriminator shape matching) ------------------

_arch("arch09_gan_generator_correct", """
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 784)
        self.relu = nn.ReLU()
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""", {"z": ("batch", 100)}, True,
    "GAN generator: latent(100) -> 256 -> 512 -> 784",
    tags=["gan"])

_arch("arch10_gan_discriminator_correct", """
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""", {"x": ("batch", 784)}, True,
    "GAN discriminator: 784 -> 512 -> 256 -> 1",
    tags=["gan"])

_arch("arch11_gan_shape_mismatch", """
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 500)
        self.relu = nn.ReLU()
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""", {"z": ("batch", 100)}, True,
    "GAN generator outputs 500 (intentional—mismatch with discriminator "
    "shows at integration level, not within this module alone)",
    tags=["gan"])

# --- 2.6  Models with intentional bugs -----------------------------------

_arch("arch12_missing_reshape", """
import torch.nn as nn
class BadModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        h = self.conv(x)
        return self.fc(h)
""", {"x": ("batch", 3, 32, 32)}, False,
    "Bug: missing flatten between Conv2d output (4D) and Linear (2D)",
    tags=["bug", "missing_reshape"])

_arch("arch13_wrong_linear_dim", """
import torch.nn as nn
class BadLinear(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 200)
        self.fc2 = nn.Linear(100, 50)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""", {"x": ("batch", 100)}, False,
    "Bug: fc1 outputs 200 but fc2 expects 100",
    tags=["bug", "wrong_dim"])

# --- 2.7  Device mismatches ----------------------------------------------

_arch("arch14_device_mismatch_model", """
import torch.nn as nn
class DeviceBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 32)
    def forward(self, x):
        h = self.fc1(x)
        h = h.cuda()
        return self.fc2(h)
""", {"x": ("batch", 64)}, False,
    "Device bug: fc1 output moved to CUDA but fc2 weights on CPU",
    tags=["device", "bug"])

_arch("arch15_device_correct_cpu", """
import torch.nn as nn
class AllCPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)
""", {"x": ("batch", 64)}, True,
    "All on CPU — no device issues",
    tags=["device"])

# --- 2.8  Phase-dependent behaviour --------------------------------------

_arch("arch16_phase_dropout_batchnorm", """
import torch.nn as nn
class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.bn = nn.BatchNorm1d(128)
        self.drop = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.bn(h)
        h = self.drop(h)
        return self.fc2(h)
""", {"x": ("batch", 128)}, True,
    "Phase-dependent: BN + Dropout behave differently in train/eval",
    tags=["phase"], default_phase=Phase.TRAIN)

_arch("arch17_phase_eval_mode", """
import torch.nn as nn
class PhaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 64)
        self.bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.bn(h)
        return self.fc2(h)
""", {"x": ("batch", 64)}, True,
    "Same model verified in EVAL phase",
    tags=["phase"], default_phase=Phase.EVAL)

# --- 2.9  More complex architectures ------------------------------------

_arch("arch18_autoencoder_correct", """
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(64, 256)
        self.dec2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        z = self.relu(self.enc2(h))
        h = self.relu(self.dec1(z))
        return self.dec2(h)
""", {"x": ("batch", 784)}, True,
    "Autoencoder: 784->256->64->256->784",
    tags=["autoencoder"])

_arch("arch19_autoencoder_bottleneck_bug", """
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(784, 256)
        self.enc2 = nn.Linear(256, 64)
        self.dec1 = nn.Linear(32, 256)
        self.dec2 = nn.Linear(256, 784)
    def forward(self, x):
        h = self.enc1(x)
        z = self.enc2(h)
        h = self.dec1(z)
        return self.dec2(h)
""", {"x": ("batch", 784)}, False,
    "Autoencoder bug: enc2 outputs 64 but dec1 expects 32",
    tags=["autoencoder", "bug"])

_arch("arch20_deep_mlp_correct", """
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""", {"x": ("batch", 256)}, True,
    "5-layer MLP, all dimensions correct",
    tags=["mlp", "deep"])


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3: CEGAR Evaluation Benchmarks
# ═══════════════════════════════════════════════════════════════════════════════

CEGAR_BENCHMARKS: List[Benchmark] = []


def _cegar(name, source, input_shapes, expect_safe, description, *, tags=None):
    CEGAR_BENCHMARKS.append(Benchmark(
        name=name, source=source, input_shapes=input_shapes,
        category=BenchmarkCategory.CEGAR_EVAL,
        expect_safe=expect_safe, description=description,
        tags=tags or [],
    ))


_cegar("cegar01_discover_single_dim", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 10)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", "features")}, True,
    "CEGAR should discover features==768")

_cegar("cegar02_discover_chain_dims", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""", {"x": ("batch", "d")}, True,
    "CEGAR discovers d==256 from fc1's in_features")

_cegar("cegar03_discover_with_relu", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc2(h)
""", {"x": ("batch", "dim")}, True,
    "CEGAR discovers dim==100 through relu (shape-preserving)")

_cegar("cegar04_real_bug_not_spurious", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 50)
        self.fc2 = nn.Linear(30, 10)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""", {"x": ("batch", 100)}, False,
    "CEGAR should identify real bug: fc1 outputs 50, fc2 expects 30")

_cegar("cegar05_multiple_symbolic", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(512, 256)
    def forward(self, x):
        return self.fc(x)
""", {"x": ("batch", "d_model")}, True,
    "CEGAR discovers d_model==512")

_cegar("cegar06_conv_symbolic_channels", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
    def forward(self, x):
        return self.conv(x)
""", {"x": ("batch", 3, "h", "w")}, True,
    "Conv2d with symbolic spatial dims, concrete channels")

_cegar("cegar07_deep_chain_discovery", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.fc4(h)
""", {"x": ("batch", "input_dim")}, True,
    "CEGAR discovers input_dim==1024 across 4-layer chain")

_cegar("cegar08_dropout_chain", """
import torch.nn as nn
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(200, 100)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(100, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = self.drop(h)
        return self.fc2(h)
""", {"x": ("batch", "n_feat")}, True,
    "CEGAR discovers n_feat==200 through dropout")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 & 5 Benchmarks: shared with PyTea + Architecture benchmarks
# Z3 stats and ablation study use the same benchmarks as sections 1-3.
# ═══════════════════════════════════════════════════════════════════════════════


# ═══════════════════════════════════════════════════════════════════════════════
# Runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_single_benchmark(bm: Benchmark) -> BenchmarkResult:
    """Run verify_model on a single benchmark and collect results."""
    t0 = time.monotonic()
    try:
        result = verify_model(
            source=bm.source,
            input_shapes=bm.input_shapes,
            default_device=bm.default_device,
            default_phase=bm.default_phase,
        )
        elapsed = (time.monotonic() - t0) * 1000

        violations = []
        violation_kinds = []
        if result.counterexample:
            violations = result.counterexample.violations
            violation_kinds = list({v.kind for v in violations})

        num_steps = result.graph.num_steps if result.graph else 0
        num_layers = len(result.graph.layers) if result.graph else 0
        cert_k = result.certificate.k if result.certificate else 0

        return BenchmarkResult(
            name=bm.name,
            category=bm.category.value,
            description=bm.description,
            expect_safe=bm.expect_safe,
            actual_safe=result.safe,
            correct=(result.safe == bm.expect_safe),
            verification_time_ms=elapsed,
            num_violations=len(violations),
            violation_kinds=violation_kinds,
            num_steps=num_steps,
            num_layers=num_layers,
            certificate_k=cert_k,
            errors=result.errors,
            pytea_info=asdict(bm.pytea_info) if bm.pytea_info else None,
            tags=bm.tags,
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return BenchmarkResult(
            name=bm.name,
            category=bm.category.value,
            description=bm.description,
            expect_safe=bm.expect_safe,
            actual_safe=True,
            correct=False,
            verification_time_ms=elapsed,
            num_violations=0,
            violation_kinds=[],
            num_steps=0,
            num_layers=0,
            certificate_k=0,
            errors=[f"Exception: {exc}"],
            pytea_info=asdict(bm.pytea_info) if bm.pytea_info else None,
            tags=bm.tags,
        )


def run_cegar_benchmark(bm: Benchmark) -> CEGARBenchmarkResult:
    """Run CEGAR on a single benchmark and collect predicate discovery info."""
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            source=bm.source,
            input_shapes=bm.input_shapes,
            max_iterations=10,
            default_device=bm.default_device,
            default_phase=bm.default_phase,
        )
        elapsed = (time.monotonic() - t0) * 1000

        predicates = [p.pretty() for p in result.discovered_predicates]
        contracts = [c.pretty() for c in result.contracts_inferred]
        real_bugs = [v.message for v in result.real_bugs]

        iter_details = []
        for rec in result.iteration_log:
            iter_details.append({
                "iteration": rec.iteration,
                "num_violations": rec.num_violations,
                "num_spurious": rec.num_spurious,
                "num_real": rec.num_real,
                "predicates_added": [p.pretty() for p in rec.predicates_added],
                "time_ms": rec.time_ms,
            })

        return CEGARBenchmarkResult(
            name=bm.name,
            description=bm.description,
            is_safe=result.is_safe,
            num_predicates_discovered=len(result.discovered_predicates),
            predicates=predicates,
            iterations=result.iterations,
            final_status=result.final_status.name,
            contracts_inferred=contracts,
            total_time_ms=elapsed,
            iteration_details=iter_details,
            real_bugs=real_bugs,
        )
    except Exception as exc:
        elapsed = (time.monotonic() - t0) * 1000
        return CEGARBenchmarkResult(
            name=bm.name,
            description=bm.description,
            is_safe=False,
            num_predicates_discovered=0,
            predicates=[],
            iterations=0,
            final_status=f"ERROR: {exc}",
            contracts_inferred=[],
            total_time_ms=elapsed,
            iteration_details=[],
            real_bugs=[str(exc)],
        )


def run_z3_stats_benchmark(bm: Benchmark) -> Z3StatsResult:
    """Run model checker with Z3 stats collection for a single benchmark."""
    if not HAS_Z3:
        return Z3StatsResult(
            name=bm.name, num_z3_queries=0, total_solve_time_ms=0.0,
            per_query_times_ms=[], num_symbolic_dims=0, num_constraints=0,
            result_safe=True,
        )

    query_times: List[float] = []
    num_queries = 0
    num_constraints = 0
    num_symbolic = 0

    try:
        graph = extract_computation_graph(bm.source)
    except (ValueError, SyntaxError):
        return Z3StatsResult(
            name=bm.name, num_z3_queries=0, total_solve_time_ms=0.0,
            per_query_times_ms=[], num_symbolic_dims=0, num_constraints=0,
            result_safe=True,
        )

    from src.model_checker import BoundedModelChecker, ConstraintVerifier, _Z3Context, ModelState

    checker = ConstraintVerifier(
        graph,
        input_shapes=bm.input_shapes,
        default_device=bm.default_device,
        default_phase=bm.default_phase,
    )

    # Count symbolic dimensions in the Z3 context
    num_symbolic = len(checker.ctx._sym_dims)

    # Run base-case and collect step-by-step Z3 queries
    states: List[ModelState] = [checker._init_state.copy()]
    all_violations = []

    for idx, step in enumerate(graph.steps[:checker.max_k]):
        current = states[-1]
        new_state, step_violations = checker._step_transition(current, step)
        all_violations.extend(step_violations)
        states.append(new_state)

    # Now run Z3 safety checks with timing per step
    if HAS_Z3:
        k0 = checker._build_kripke_state(0, checker._init_state)
        kripke_states = [k0]
        for idx in range(min(len(states) - 1, len(graph.steps))):
            step = graph.steps[idx]
            s_pre = states[idx]
            cur_k = kripke_states[-1]

            # Collect safety constraints from all domains
            symbolic_constraints = []
            symbolic_constraints.extend(
                checker._encode_shape_safety(cur_k, step, s_pre, idx))
            symbolic_constraints.extend(
                checker._encode_device_safety(cur_k, step, s_pre, idx))

            # Build post Kripke state for next iteration
            post_model = states[idx + 1] if idx + 1 < len(states) else s_pre
            post_k = checker._build_kripke_state(idx + 1, post_model)
            kripke_states.append(post_k)

            if not symbolic_constraints:
                continue

            solver = z3.Solver()
            solver.set("timeout", 5000)

            for c in checker.ctx.positive_dim_constraints():
                solver.add(c)

            neg = z3.Not(z3.And(*symbolic_constraints))
            solver.add(neg)

            num_constraints += len(symbolic_constraints)

            qt0 = time.monotonic()
            solver.check()
            qt1 = time.monotonic()

            query_times.append((qt1 - qt0) * 1000)
            num_queries += 1

            # Update symbolic dim count after queries
            num_symbolic = len(checker.ctx._sym_dims)

    is_safe = len(all_violations) == 0

    return Z3StatsResult(
        name=bm.name,
        num_z3_queries=num_queries,
        total_solve_time_ms=sum(query_times),
        per_query_times_ms=query_times,
        num_symbolic_dims=num_symbolic,
        num_constraints=num_constraints,
        result_safe=is_safe,
    )


def run_ablation_benchmark(bm: Benchmark) -> AblationResult:
    """Run ablation study for a single benchmark: shape-only, model-checker,
    full pipeline, device on/off."""

    # --- A) Shape analysis alone (no model checking) ----------------------
    t0 = time.monotonic()
    try:
        shape_result = analyze_shapes(bm.source)
        shape_safe = len(shape_result.errors) == 0
        shape_errors = len(shape_result.errors)
    except Exception:
        shape_safe = True
        shape_errors = 0
    shape_time = (time.monotonic() - t0) * 1000

    # --- B) Model checking without CEGAR ---------------------------------
    t0 = time.monotonic()
    try:
        mc_result = verify_model(
            source=bm.source,
            input_shapes=bm.input_shapes,
            default_device=bm.default_device,
            default_phase=bm.default_phase,
        )
        mc_safe = mc_result.safe
        mc_errors = len(mc_result.counterexample.violations) if mc_result.counterexample else 0
    except Exception:
        mc_safe = True
        mc_errors = 0
    mc_time = (time.monotonic() - t0) * 1000

    # --- C) Full pipeline: model checking + CEGAR -------------------------
    t0 = time.monotonic()
    try:
        cegar_result = run_shape_cegar(
            source=bm.source,
            input_shapes=bm.input_shapes,
            max_iterations=10,
            default_device=bm.default_device,
            default_phase=bm.default_phase,
        )
        full_safe = cegar_result.is_safe
        full_predicates = len(cegar_result.discovered_predicates)
        full_iterations = cegar_result.iterations
    except Exception:
        full_safe = True
        full_predicates = 0
        full_iterations = 0
    full_time = (time.monotonic() - t0) * 1000

    # --- D) Device tracking ON (default_device=CPU, normal) ---------------
    t0 = time.monotonic()
    try:
        dev_on_result = verify_model(
            source=bm.source,
            input_shapes=bm.input_shapes,
            default_device=Device.CPU,
            default_phase=bm.default_phase,
        )
        dev_on_safe = dev_on_result.safe
        dev_on_violations = (
            len(dev_on_result.counterexample.violations)
            if dev_on_result.counterexample else 0
        )
    except Exception:
        dev_on_safe = True
        dev_on_violations = 0
    dev_on_time = (time.monotonic() - t0) * 1000

    # --- E) Device tracking OFF: run with all on same device ---------------
    # We simulate "device OFF" by NOT using .cuda() — just re-verify on CPU
    # with the same source but comparing to the ON case.
    t0 = time.monotonic()
    try:
        # Use CUDA as default so .cuda() is a no-op (both on same device)
        dev_off_result = verify_model(
            source=bm.source,
            input_shapes=bm.input_shapes,
            default_device=Device.CUDA_0,
            default_phase=bm.default_phase,
        )
        dev_off_safe = dev_off_result.safe
        dev_off_violations = (
            len(dev_off_result.counterexample.violations)
            if dev_off_result.counterexample else 0
        )
    except Exception:
        dev_off_safe = True
        dev_off_violations = 0
    dev_off_time = (time.monotonic() - t0) * 1000

    return AblationResult(
        name=bm.name,
        shape_only_safe=shape_safe,
        shape_only_time_ms=shape_time,
        shape_only_errors=shape_errors,
        model_checker_safe=mc_safe,
        model_checker_time_ms=mc_time,
        model_checker_errors=mc_errors,
        full_pipeline_safe=full_safe,
        full_pipeline_time_ms=full_time,
        full_pipeline_predicates=full_predicates,
        full_pipeline_iterations=full_iterations,
        device_on_safe=dev_on_safe,
        device_on_time_ms=dev_on_time,
        device_on_violations=dev_on_violations,
        device_off_safe=dev_off_safe,
        device_off_time_ms=dev_off_time,
        device_off_violations=dev_off_violations,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# Reporting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def print_section_header(title: str):
    width = 78
    print()
    print("═" * width)
    print(f"  {title}")
    print("═" * width)


def print_pytea_comparison_table(results: List[BenchmarkResult]):
    """Print a table comparing TensorGuard vs PyTea capabilities."""
    print_section_header("PyTea Comparison Summary")

    pytea_can_count = sum(
        1 for r in results if r.pytea_info and r.pytea_info.get("can_handle")
    )
    pytea_cannot_count = sum(
        1 for r in results if r.pytea_info and not r.pytea_info.get("can_handle")
    )
    tensorguard_correct = sum(1 for r in results if r.correct)

    print(f"\n  Total benchmarks:        {len(results)}")
    print(f"  PyTea can handle:        {pytea_can_count}")
    print(f"  PyTea CANNOT handle:     {pytea_cannot_count}")
    print(f"  TensorGuard correct:        {tensorguard_correct}/{len(results)}")
    print(f"  TensorGuard accuracy:       "
          f"{tensorguard_correct / len(results) * 100:.1f}%")

    print(f"\n  {'Name':<40} {'Expect':>6} {'Got':>6} {'OK':>3} "
          f"{'PyTea':>6} {'Time':>8}")
    print("  " + "─" * 73)
    for r in results:
        exp = "safe" if r.expect_safe else "UNSAFE"
        got = "safe" if r.actual_safe else "UNSAFE"
        ok = "✓" if r.correct else "✗"
        pt = "✓" if (r.pytea_info and r.pytea_info.get("can_handle")) else "✗"
        print(f"  {r.name:<40} {exp:>6} {got:>6} {ok:>3} {pt:>6} "
              f"{r.verification_time_ms:>7.1f}ms")

    # Breakdown by PyTea category
    print(f"\n  --- Breakdown by PyTea capability ---")
    # Group by whether PyTea can handle
    can_results = [r for r in results
                   if r.pytea_info and r.pytea_info.get("can_handle")]
    cannot_results = [r for r in results
                      if r.pytea_info and not r.pytea_info.get("can_handle")]

    if can_results:
        can_correct = sum(1 for r in can_results if r.correct)
        print(f"  PyTea CAN handle ({len(can_results)}):  "
              f"TensorGuard {can_correct}/{len(can_results)} correct")
    if cannot_results:
        cannot_correct = sum(1 for r in cannot_results if r.correct)
        print(f"  PyTea CANNOT handle ({len(cannot_results)}): "
              f"TensorGuard {cannot_correct}/{len(cannot_results)} correct")
        print(f"    Features PyTea lacks:")
        reasons = sorted(set(
            r.pytea_info["reason"] for r in cannot_results if r.pytea_info
        ))
        for reason in reasons:
            print(f"      • {reason}")


def print_architecture_table(results: List[BenchmarkResult]):
    """Print results table for real-world architectures."""
    print_section_header("Real-World Architecture Results")

    correct = sum(1 for r in results if r.correct)
    print(f"\n  Total: {len(results)}, Correct: {correct}/{len(results)} "
          f"({correct / len(results) * 100:.1f}%)")
    print(f"\n  {'Name':<42} {'Expect':>6} {'Got':>6} {'OK':>3} "
          f"{'Steps':>5} {'Layers':>6} {'Time':>8}")
    print("  " + "─" * 80)
    for r in results:
        exp = "safe" if r.expect_safe else "UNSAFE"
        got = "safe" if r.actual_safe else "UNSAFE"
        ok = "✓" if r.correct else "✗"
        print(f"  {r.name:<42} {exp:>6} {got:>6} {ok:>3} "
              f"{r.num_steps:>5} {r.num_layers:>6} "
              f"{r.verification_time_ms:>7.1f}ms")

    # Group by tag
    tag_counts: Dict[str, Tuple[int, int]] = {}  # tag -> (total, correct)
    for r in results:
        for tag in r.tags:
            tot, corr = tag_counts.get(tag, (0, 0))
            tag_counts[tag] = (tot + 1, corr + (1 if r.correct else 0))
    if tag_counts:
        print(f"\n  --- By tag ---")
        for tag, (tot, corr) in sorted(tag_counts.items()):
            print(f"    {tag:<20} {corr}/{tot}")


def print_cegar_table(results: List[CEGARBenchmarkResult]):
    """Print CEGAR evaluation results."""
    print_section_header("CEGAR Evaluation Results")

    total_preds = sum(r.num_predicates_discovered for r in results)
    total_contracts = sum(len(r.contracts_inferred) for r in results)
    avg_iters = (
        sum(r.iterations for r in results) / len(results) if results else 0
    )

    print(f"\n  Total benchmarks:        {len(results)}")
    print(f"  Total predicates found:  {total_preds}")
    print(f"  Total contracts inferred:{total_contracts}")
    print(f"  Avg iterations:          {avg_iters:.1f}")

    print(f"\n  {'Name':<35} {'Safe':>4} {'Preds':>5} {'Iters':>5} "
          f"{'Status':<15} {'Time':>8}")
    print("  " + "─" * 78)
    for r in results:
        safe = "✓" if r.is_safe else "✗"
        print(f"  {r.name:<35} {safe:>4} {r.num_predicates_discovered:>5} "
              f"{r.iterations:>5} {r.final_status:<15} "
              f"{r.total_time_ms:>7.1f}ms")

    # Show discovered predicates
    print(f"\n  --- Discovered Predicates ---")
    for r in results:
        if r.predicates:
            print(f"  {r.name}:")
            for p in r.predicates:
                print(f"    • {p}")

    # Show inferred contracts
    print(f"\n  --- Inferred Contracts ---")
    for r in results:
        if r.contracts_inferred:
            print(f"  {r.name}:")
            for c in r.contracts_inferred:
                print(f"    → {c}")

    # Show iteration convergence
    print(f"\n  --- Convergence Details ---")
    for r in results:
        if r.iteration_details:
            print(f"  {r.name}: {r.iterations} iterations")
            for it in r.iteration_details:
                preds_str = ", ".join(it["predicates_added"]) or "none"
                print(f"    iter {it['iteration']}: "
                      f"{it['num_violations']} violations, "
                      f"{it['num_spurious']} spurious, "
                      f"{it['num_real']} real, "
                      f"preds=[{preds_str}] "
                      f"({it['time_ms']:.1f}ms)")


def print_z3_stats_table(results: List[Z3StatsResult]):
    """Print Z3 solver statistics."""
    print_section_header("Z3 Solver Statistics")

    total_queries = sum(r.num_z3_queries for r in results)
    total_time = sum(r.total_solve_time_ms for r in results)
    all_times = [t for r in results for t in r.per_query_times_ms]

    print(f"\n  Total benchmarks:      {len(results)}")
    print(f"  Total Z3 queries:      {total_queries}")
    print(f"  Total Z3 solve time:   {total_time:.1f}ms")
    if all_times:
        print(f"  Avg query time:        {sum(all_times) / len(all_times):.2f}ms")
        print(f"  Min query time:        {min(all_times):.2f}ms")
        print(f"  Max query time:        {max(all_times):.2f}ms")
        # Distribution
        under_1 = sum(1 for t in all_times if t < 1.0)
        under_10 = sum(1 for t in all_times if 1.0 <= t < 10.0)
        under_100 = sum(1 for t in all_times if 10.0 <= t < 100.0)
        over_100 = sum(1 for t in all_times if t >= 100.0)
        print(f"  Distribution:          <1ms: {under_1}, "
              f"1-10ms: {under_10}, 10-100ms: {under_100}, "
              f">100ms: {over_100}")

    print(f"\n  {'Name':<40} {'Queries':>7} {'SymDims':>7} "
          f"{'Constrs':>8} {'Time':>8} {'Safe':>4}")
    print("  " + "─" * 78)
    for r in results:
        safe = "✓" if r.result_safe else "✗"
        print(f"  {r.name:<40} {r.num_z3_queries:>7} "
              f"{r.num_symbolic_dims:>7} {r.num_constraints:>8} "
              f"{r.total_solve_time_ms:>7.1f}ms {safe:>4}")


def print_ablation_table(results: List[AblationResult]):
    """Print ablation study results."""
    print_section_header("Ablation Study")

    print(f"\n  {'Name':<30} │ {'Shape':>10} │ {'MC':>10} │ "
          f"{'MC+CEGAR':>10} │ {'Dev ON':>10} │ {'Dev OFF':>10}")
    print("  " + "─" * 95)
    for r in results:
        s_lbl = f"{'✓' if r.shape_only_safe else '✗'} {r.shape_only_time_ms:.0f}ms"
        m_lbl = f"{'✓' if r.model_checker_safe else '✗'} {r.model_checker_time_ms:.0f}ms"
        f_lbl = (f"{'✓' if r.full_pipeline_safe else '✗'} "
                 f"{r.full_pipeline_time_ms:.0f}ms")
        d_on = f"{'✓' if r.device_on_safe else '✗'} {r.device_on_violations}v"
        d_off = f"{'✓' if r.device_off_safe else '✗'} {r.device_off_violations}v"
        print(f"  {r.name:<30} │ {s_lbl:>10} │ {m_lbl:>10} │ "
              f"{f_lbl:>10} │ {d_on:>10} │ {d_off:>10}")

    # Aggregate statistics
    print(f"\n  --- Aggregate ---")
    n = len(results)
    if n == 0:
        return

    print(f"  Shape-only avg time:     "
          f"{sum(r.shape_only_time_ms for r in results) / n:.1f}ms")
    print(f"  Model-checker avg time:  "
          f"{sum(r.model_checker_time_ms for r in results) / n:.1f}ms")
    print(f"  Full pipeline avg time:  "
          f"{sum(r.full_pipeline_time_ms for r in results) / n:.1f}ms")
    print(f"  CEGAR avg predicates:    "
          f"{sum(r.full_pipeline_predicates for r in results) / n:.1f}")
    print(f"  CEGAR avg iterations:    "
          f"{sum(r.full_pipeline_iterations for r in results) / n:.1f}")

    # Device tracking comparison
    dev_diff = sum(
        1 for r in results
        if r.device_on_safe != r.device_off_safe
    )
    print(f"  Device tracking changes:  {dev_diff}/{n} benchmarks differ")


# ═══════════════════════════════════════════════════════════════════════════════
# Main entry point
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 78)
    print("  TensorGuard Model Checker Evaluation Suite")
    print("=" * 78)
    print(f"  Z3 available: {HAS_Z3}")
    print(f"  PyTea comparison benchmarks: {len(PYTEA_BENCHMARKS)}")
    print(f"  Architecture benchmarks:     {len(ARCH_BENCHMARKS)}")
    print(f"  CEGAR benchmarks:            {len(CEGAR_BENCHMARKS)}")

    all_results: Dict[str, Any] = {
        "meta": {
            "z3_available": HAS_Z3,
            "num_pytea_benchmarks": len(PYTEA_BENCHMARKS),
            "num_arch_benchmarks": len(ARCH_BENCHMARKS),
            "num_cegar_benchmarks": len(CEGAR_BENCHMARKS),
        },
    }

    # ── Section 1: PyTea Comparison ────────────────────────────────────────
    print(f"\n  Running {len(PYTEA_BENCHMARKS)} PyTea comparison benchmarks...")
    pytea_results: List[BenchmarkResult] = []
    for i, bm in enumerate(PYTEA_BENCHMARKS, 1):
        r = run_single_benchmark(bm)
        status = "✓" if r.correct else "✗"
        print(f"    [{i:>2}/{len(PYTEA_BENCHMARKS)}] {status} {bm.name}")
        pytea_results.append(r)

    print_pytea_comparison_table(pytea_results)
    all_results["pytea_comparison"] = {
        "total": len(pytea_results),
        "correct": sum(1 for r in pytea_results if r.correct),
        "accuracy": (sum(1 for r in pytea_results if r.correct)
                     / len(pytea_results) * 100 if pytea_results else 0),
        "pytea_can_handle": sum(
            1 for r in pytea_results
            if r.pytea_info and r.pytea_info.get("can_handle")
        ),
        "pytea_cannot_handle": sum(
            1 for r in pytea_results
            if r.pytea_info and not r.pytea_info.get("can_handle")
        ),
        "total_time_ms": sum(r.verification_time_ms for r in pytea_results),
        "results": [asdict(r) for r in pytea_results],
    }

    # ── Section 2: Real-World Architectures ────────────────────────────────
    print(f"\n  Running {len(ARCH_BENCHMARKS)} architecture benchmarks...")
    arch_results: List[BenchmarkResult] = []
    for i, bm in enumerate(ARCH_BENCHMARKS, 1):
        r = run_single_benchmark(bm)
        status = "✓" if r.correct else "✗"
        print(f"    [{i:>2}/{len(ARCH_BENCHMARKS)}] {status} {bm.name}")
        arch_results.append(r)

    print_architecture_table(arch_results)
    all_results["architecture"] = {
        "total": len(arch_results),
        "correct": sum(1 for r in arch_results if r.correct),
        "accuracy": (sum(1 for r in arch_results if r.correct)
                     / len(arch_results) * 100 if arch_results else 0),
        "total_time_ms": sum(r.verification_time_ms for r in arch_results),
        "results": [asdict(r) for r in arch_results],
    }

    # ── Section 3: CEGAR Evaluation ───────────────────────────────────────
    print(f"\n  Running {len(CEGAR_BENCHMARKS)} CEGAR benchmarks...")
    cegar_results: List[CEGARBenchmarkResult] = []
    for i, bm in enumerate(CEGAR_BENCHMARKS, 1):
        r = run_cegar_benchmark(bm)
        status = "✓" if r.is_safe else "✗"
        print(f"    [{i:>2}/{len(CEGAR_BENCHMARKS)}] {status} {bm.name} "
              f"({r.num_predicates_discovered} preds, "
              f"{r.iterations} iters)")
        cegar_results.append(r)

    print_cegar_table(cegar_results)
    all_results["cegar_evaluation"] = {
        "total": len(cegar_results),
        "total_predicates": sum(
            r.num_predicates_discovered for r in cegar_results
        ),
        "total_contracts": sum(
            len(r.contracts_inferred) for r in cegar_results
        ),
        "avg_iterations": (
            sum(r.iterations for r in cegar_results) / len(cegar_results)
            if cegar_results else 0
        ),
        "total_time_ms": sum(r.total_time_ms for r in cegar_results),
        "results": [asdict(r) for r in cegar_results],
    }

    # ── Section 4: Z3 Solver Statistics ───────────────────────────────────
    # Use all benchmarks that have symbolic dimensions
    z3_benchmarks = (
        [b for b in PYTEA_BENCHMARKS if any(
            isinstance(d, str) for shape in b.input_shapes.values() for d in shape
        )]
        + [b for b in ARCH_BENCHMARKS if any(
            isinstance(d, str) for shape in b.input_shapes.values() for d in shape
        )]
        + CEGAR_BENCHMARKS
    )
    # Also include some concrete-dim benchmarks for comparison
    concrete_sample = [b for b in PYTEA_BENCHMARKS if not any(
        isinstance(d, str) for shape in b.input_shapes.values() for d in shape
    )][:5]
    z3_benchmarks += concrete_sample

    print(f"\n  Running Z3 stats on {len(z3_benchmarks)} benchmarks...")
    z3_results: List[Z3StatsResult] = []
    for i, bm in enumerate(z3_benchmarks, 1):
        r = run_z3_stats_benchmark(bm)
        print(f"    [{i:>2}/{len(z3_benchmarks)}] {bm.name} "
              f"({r.num_z3_queries} queries, {r.total_solve_time_ms:.1f}ms)")
        z3_results.append(r)

    print_z3_stats_table(z3_results)
    all_results["z3_statistics"] = {
        "total_benchmarks": len(z3_results),
        "total_queries": sum(r.num_z3_queries for r in z3_results),
        "total_solve_time_ms": sum(r.total_solve_time_ms for r in z3_results),
        "per_query_times_ms": [t for r in z3_results for t in r.per_query_times_ms],
        "results": [asdict(r) for r in z3_results],
    }

    # ── Section 5: Ablation Study ─────────────────────────────────────────
    # Select representative benchmarks across categories
    ablation_benchmarks = (
        PYTEA_BENCHMARKS[:6]      # basic shapes
        + [b for b in PYTEA_BENCHMARKS if "device" in b.tags][:3]
        + [b for b in PYTEA_BENCHMARKS if "phase" in b.tags][:2]
        + ARCH_BENCHMARKS[:6]     # architectures
        + [b for b in ARCH_BENCHMARKS if "device" in b.tags][:2]
        + CEGAR_BENCHMARKS[:4]    # CEGAR benchmarks
    )
    # Deduplicate
    seen_names: set = set()
    unique_ablation: List[Benchmark] = []
    for b in ablation_benchmarks:
        if b.name not in seen_names:
            seen_names.add(b.name)
            unique_ablation.append(b)

    print(f"\n  Running ablation study on {len(unique_ablation)} benchmarks...")
    ablation_results: List[AblationResult] = []
    for i, bm in enumerate(unique_ablation, 1):
        r = run_ablation_benchmark(bm)
        print(f"    [{i:>2}/{len(unique_ablation)}] {bm.name}")
        ablation_results.append(r)

    print_ablation_table(ablation_results)
    all_results["ablation_study"] = {
        "total": len(ablation_results),
        "results": [asdict(r) for r in ablation_results],
        "aggregate": {
            "shape_only_avg_time_ms": (
                sum(r.shape_only_time_ms for r in ablation_results)
                / len(ablation_results) if ablation_results else 0
            ),
            "model_checker_avg_time_ms": (
                sum(r.model_checker_time_ms for r in ablation_results)
                / len(ablation_results) if ablation_results else 0
            ),
            "full_pipeline_avg_time_ms": (
                sum(r.full_pipeline_time_ms for r in ablation_results)
                / len(ablation_results) if ablation_results else 0
            ),
            "cegar_avg_predicates": (
                sum(r.full_pipeline_predicates for r in ablation_results)
                / len(ablation_results) if ablation_results else 0
            ),
            "device_tracking_differs": sum(
                1 for r in ablation_results
                if r.device_on_safe != r.device_off_safe
            ),
        },
    }

    # ── Save all results ──────────────────────────────────────────────────
    with open(OUTPUT_FILE, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print_section_header("Final Summary")
    total_benchmarks = (
        len(pytea_results) + len(arch_results) + len(cegar_results)
        + len(z3_results) + len(ablation_results)
    )
    total_correct = (
        sum(1 for r in pytea_results if r.correct)
        + sum(1 for r in arch_results if r.correct)
    )
    total_checked = len(pytea_results) + len(arch_results)
    print(f"\n  Total benchmark runs:         {total_benchmarks}")
    print(f"  Verification accuracy:        "
          f"{total_correct}/{total_checked} "
          f"({total_correct / total_checked * 100:.1f}%)"
          if total_checked else "  No verification benchmarks")
    print(f"  Total predicates discovered:  "
          f"{sum(r.num_predicates_discovered for r in cegar_results)}")
    print(f"  Total Z3 queries:             "
          f"{sum(r.num_z3_queries for r in z3_results)}")
    print(f"\n  Results saved to: {OUTPUT_FILE}")
    print("═" * 78)


if __name__ == "__main__":
    main()
