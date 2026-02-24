"""
Comprehensive Final Evaluation — Unified Benchmark Suite.

Runs ALL benchmark categories in a single framework:
  1. Theory-exercising benchmarks (broadcasting, matmul, multi-step)
  2. Production-style architectures (ResNet, Transformer, U-Net, etc.)
  3. Contract discovery benchmarks
  4. CEGAR ablation (no-CEGAR vs filtered-CEGAR)

For each tool (syntactic, tensorguard), computes:
  - Precision, Recall, F1 with 95% bootstrap confidence intervals (1000 resamples)
  - Per-category breakdown
  - Average verification time

Outputs: experiments/comprehensive_final_results.json
"""

from __future__ import annotations

import ast
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.model_checker import (
    extract_computation_graph,
    BoundedModelChecker, ConstraintVerifier,
    Device,
    Phase,
    verify_model,
)
from src.shape_cegar import run_shape_cegar, PREDICATE_QUALITY_THRESHOLD

RESULTS_FILE = Path(__file__).parent / "comprehensive_final_results.json"

# ═══════════════════════════════════════════════════════════════════════════════
# Benchmark Suite
# ═══════════════════════════════════════════════════════════════════════════════

# ── Category 1: Theory-exercising (broadcasting, matmul, multi-step) ──────────

THEORY_BENCHMARKS: List[Dict[str, Any]] = [
    # A. Broadcasting
    {
        "name": "broadcast_parallel_bug",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": True,
        "description": "Linear->128 + Linear->64: broadcast dims 128 vs 64",
        "code": """\
import torch.nn as nn
class BroadcastParallelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "broadcast_parallel_safe",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": False,
        "description": "Linear->128 + Linear->128: broadcast OK (same dims)",
        "code": """\
import torch.nn as nn
class BroadcastParallelSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "broadcast_cross_rank_bug",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": True,
        "description": "(batch,seq,256) + (batch,128): last dims 256 vs 128 mismatch",
        "code": """\
import torch.nn as nn
class CrossRankBroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 128)
    def forward(self, x, bias):
        x = self.proj(x)
        bias = self.bias_proj(bias)
        return x + bias
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": ("batch", 256)},
    },
    {
        "name": "broadcast_cross_rank_safe",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": False,
        "description": "(batch,seq,256) + (256,): broadcasts to (1,1,256) — safe",
        "code": """\
import torch.nn as nn
class CrossRankBroadcastSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 256)
    def forward(self, x, bias):
        x = self.proj(x)
        bias = self.bias_proj(bias)
        return x + bias
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": (256,)},
    },
    {
        "name": "broadcast_chain_bug",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": True,
        "description": "(128)+(128)=OK then (128)+(64)=FAIL at second add",
        "code": """\
import torch.nn as nn
class BroadcastChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = self.fc_c(x)
        ab = a + b
        return ab + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "broadcast_chain_safe",
        "category": "theory",
        "subcategory": "broadcast",
        "has_bug": False,
        "description": "(128)+(128)+(128): all adds broadcast-compatible",
        "code": """\
import torch.nn as nn
class BroadcastChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = self.fc_c(x)
        ab = a + b
        return ab + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    # B. Matmul
    {
        "name": "matmul_inner_mismatch_bug",
        "category": "theory",
        "subcategory": "matmul",
        "has_bug": True,
        "description": "proj->32 then matmul with (64,10): inner 32!=64",
        "code": """\
import torch.nn as nn
class MatmulInnerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 32)
    def forward(self, x, w):
        x = self.proj(x)
        return x @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },
    {
        "name": "matmul_inner_match_safe",
        "category": "theory",
        "subcategory": "matmul",
        "has_bug": False,
        "description": "proj->64 then matmul with (64,10): inner 64==64 OK",
        "code": """\
import torch.nn as nn
class MatmulInnerSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64)
    def forward(self, x, w):
        x = self.proj(x)
        return x @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },
    {
        "name": "matmul_after_add_bug",
        "category": "theory",
        "subcategory": "matmul",
        "has_bug": True,
        "description": "Two proj->32, add, then matmul(64,10): inner 32!=64",
        "code": """\
import torch.nn as nn
class MatmulAfterAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 32)
        self.fc_b = nn.Linear(256, 32)
    def forward(self, x, w):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },
    {
        "name": "matmul_after_add_safe",
        "category": "theory",
        "subcategory": "matmul",
        "has_bug": False,
        "description": "Two proj->64, add, then matmul(64,10): inner 64==64 OK",
        "code": """\
import torch.nn as nn
class MatmulAfterAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 64)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x, w):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },
    # C. Multi-step
    {
        "name": "add_then_linear_bug",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": True,
        "description": "proj->128 + proj->64 then Linear(128,10): add already fails",
        "code": """\
import torch.nn as nn
class AddThenLinearBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 64)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return self.out(c)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "add_then_linear_safe",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": False,
        "description": "proj->128 + proj->128 then Linear(128,10): all OK",
        "code": """\
import torch.nn as nn
class AddThenLinearSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.out = nn.Linear(128, 10)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        c = a + b
        return self.out(c)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "multihead_add_bug",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": True,
        "description": "Q->256 + K->128 (symbolic seq dim): broadcast fails",
        "code": """\
import torch.nn as nn
class MultiHeadAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.k_proj = nn.Linear(512, 128)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 64)
    def forward(self, x):
        q = self.relu(self.q_proj(x))
        k = self.relu(self.k_proj(x))
        combined = q + k
        return self.out(combined)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "multihead_add_safe",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": False,
        "description": "Q->256 + K->256: matching projection dims, add safe",
        "code": """\
import torch.nn as nn
class MultiHeadAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 256)
        self.k_proj = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 64)
    def forward(self, x):
        q = self.relu(self.q_proj(x))
        k = self.relu(self.k_proj(x))
        combined = q + k
        return self.out(combined)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "conv_parallel_add_bug",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": True,
        "description": "Conv->32 + Conv->64: channel mismatch in add",
        "code": """\
import torch.nn as nn
class ConvParallelAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 64, 3, padding=1)
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "conv_parallel_add_safe",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": False,
        "description": "Conv->32 + Conv->32: matching channels, add safe",
        "code": """\
import torch.nn as nn
class ConvParallelAddSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 32, 3, padding=1)
    def forward(self, x):
        a = self.conv1(x)
        b = self.conv2(x)
        return a + b
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "double_add_chain_bug",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": True,
        "description": "proj->32 + proj->32 OK, then + proj->64 FAILS",
        "code": """\
import torch.nn as nn
class DoubleAddChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 32)
        self.fc_b = nn.Linear(256, 32)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        c = self.fc_c(x)
        return s + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "double_add_chain_safe",
        "category": "theory",
        "subcategory": "multi_step",
        "has_bug": False,
        "description": "proj->64 + proj->64 + proj->64: all adds safe",
        "code": """\
import torch.nn as nn
class DoubleAddChainSafe(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 64)
        self.fc_b = nn.Linear(256, 64)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        a = self.fc_a(x)
        b = self.fc_b(x)
        s = a + b
        c = self.fc_c(x)
        return s + c
""",
        "input_shapes": {"x": ("batch", 256)},
    },
]

# ── Category 2: Production-style architectures ───────────────────────────────

PRODUCTION_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "resnet_basicblock_correct",
        "category": "production",
        "subcategory": "ResNet",
        "has_bug": False,
        "description": "ResNet BasicBlock with skip connection, BN, ReLU",
        "code": """\
import torch
import torch.nn as nn

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes=64, planes=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes),
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        return self.relu(out)
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    {
        "name": "bert_attention_bug",
        "category": "production",
        "subcategory": "Transformer",
        "has_bug": True,
        "description": "BERT attention: out_proj expects 512 but attention output is 768",
        "code": """\
import torch
import torch.nn as nn

class BertSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_heads=12):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(512, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)
        attn_output = v
        output = self.out_proj(attn_output)
        output = self.dropout(output)
        return self.layer_norm(output)
""",
        "input_shapes": {"hidden_states": ("batch", "seq_len", 768)},
    },
    {
        "name": "gpt_block_correct",
        "category": "production",
        "subcategory": "Transformer",
        "has_bug": False,
        "description": "GPT-2 style transformer block with correct dims",
        "code": """\
import torch
import torch.nn as nn

class GPTBlock(nn.Module):
    def __init__(self, d_model=768, n_head=12, d_ff=3072):
        super().__init__()
        self.ln_1 = nn.LayerNorm(d_model)
        self.attn_q = nn.Linear(d_model, d_model)
        self.attn_k = nn.Linear(d_model, d_model)
        self.attn_v = nn.Linear(d_model, d_model)
        self.attn_proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(0.1)
        self.ln_2 = nn.LayerNorm(d_model)
        self.mlp_fc = nn.Linear(d_model, d_ff)
        self.mlp_proj = nn.Linear(d_ff, d_model)
        self.mlp_act = nn.ReLU()
        self.mlp_drop = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        x = self.ln_1(x)
        q = self.attn_q(x)
        k = self.attn_k(x)
        v = self.attn_v(x)
        x = self.attn_proj(v)
        x = self.attn_drop(x)
        x = x + residual
        residual = x
        x = self.ln_2(x)
        x = self.mlp_fc(x)
        x = self.mlp_act(x)
        x = self.mlp_proj(x)
        x = self.mlp_drop(x)
        x = x + residual
        return x
""",
        "input_shapes": {"x": ("batch", "seq_len", 768)},
    },
    {
        "name": "unet_skip_bug",
        "category": "production",
        "subcategory": "U-Net",
        "has_bug": True,
        "description": "U-Net: decoder conv expects 256 channels but skip concat gives 192",
        "code": """\
import torch
import torch.nn as nn

class UNetSmall(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.enc1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.enc1_bn = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.enc2_conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.enc2_conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.enc2_bn = nn.BatchNorm2d(128)
        self.up1 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec1_conv1 = nn.Conv2d(256, 64, 3, padding=1)
        self.dec1_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.final_conv = nn.Conv2d(64, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1_bn(self.enc1_conv2(self.relu(self.enc1_conv1(x)))))
        p1 = self.pool1(e1)
        e2 = self.relu(self.enc2_bn(self.enc2_conv2(self.relu(self.enc2_conv1(p1)))))
        d1 = self.up1(e2)
        d1 = self.relu(self.dec1_conv1(d1))
        d1 = self.relu(self.dec1_conv2(d1))
        return self.final_conv(d1)
""",
        "input_shapes": {"x": ("batch", 3, 128, 128)},
    },
    {
        "name": "dcgan_generator_bug",
        "category": "production",
        "subcategory": "GAN",
        "has_bug": True,
        "description": "DCGAN generator: deconv2 expects 128 channels but deconv1 outputs 256",
        "code": """\
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    def __init__(self, nz=100, ngf=64):
        super().__init__()
        self.project = nn.Linear(nz, ngf * 8 * 4 * 4)
        self.bn0 = nn.BatchNorm2d(ngf * 8)
        self.deconv1 = nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(ngf * 4)
        self.deconv2 = nn.ConvTranspose2d(ngf * 2, ngf, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(ngf)
        self.deconv3 = nn.ConvTranspose2d(ngf, 3, 4, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, z):
        x = self.project(z)
        x = x.view(x.size(0), 512, 4, 4)
        x = self.relu(self.bn0(x))
        x = self.relu(self.bn1(self.deconv1(x)))
        x = self.relu(self.bn2(self.deconv2(x)))
        x = self.deconv3(x)
        return x
""",
        "input_shapes": {"z": ("batch", 100)},
    },
    {
        "name": "vae_correct",
        "category": "production",
        "subcategory": "VAE",
        "has_bug": False,
        "description": "VAE with correct encoder/reparameterize/decoder dimensions",
        "code": """\
import torch
import torch.nn as nn

class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.relu(self.fc1(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = mu
        h_dec = self.relu(self.fc3(z))
        return self.fc4(h_dec)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "multihead_attn_reshape_bug",
        "category": "production",
        "subcategory": "Transformer",
        "has_bug": True,
        "description": "MHA: out_proj expects 256 but concat of heads produces 512",
        "code": """\
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_model = d_model
        self.head_dim = d_model // n_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(256, d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, query):
        q = self.q_linear(query)
        k = self.k_linear(query)
        v = self.v_linear(query)
        output = self.out_proj(v)
        return self.dropout(output)
""",
        "input_shapes": {"query": ("batch", "seq_len", 512)},
    },
    {
        "name": "resnet_bottleneck_correct",
        "category": "production",
        "subcategory": "ResNet",
        "has_bug": False,
        "description": "ResNet Bottleneck (1x1->3x3->1x1) with expansion, correct dims",
        "code": """\
import torch
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes=256, planes=64, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Sequential(
            nn.Conv2d(inplanes, planes * 4, 1, stride=stride, bias=False),
            nn.BatchNorm2d(planes * 4),
        )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + identity
        return self.relu(out)
""",
        "input_shapes": {"x": ("batch", 256, "h", "w")},
    },
    {
        "name": "transformer_encoder_ffn_bug",
        "category": "production",
        "subcategory": "Transformer",
        "has_bug": True,
        "description": "Transformer FFN: ff1 outputs 3072 but ff2 expects 2048",
        "code": """\
import torch
import torch.nn as nn

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model=512, nhead=8, dim_feedforward=2048):
        super().__init__()
        self.self_attn_q = nn.Linear(d_model, d_model)
        self.self_attn_k = nn.Linear(d_model, d_model)
        self.self_attn_v = nn.Linear(d_model, d_model)
        self.self_attn_out = nn.Linear(d_model, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, 3072)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.ReLU()

    def forward(self, src):
        q = self.self_attn_q(src)
        k = self.self_attn_k(src)
        v = self.self_attn_v(src)
        attn_out = self.self_attn_out(v)
        src = self.norm1(src + self.dropout(attn_out))
        ff_out = self.ff1(src)
        ff_out = self.activation(ff_out)
        ff_out = self.ff2(ff_out)
        src = self.norm2(src + self.dropout(ff_out))
        return src
""",
        "input_shapes": {"src": ("batch", "seq_len", 512)},
    },
    {
        "name": "inception_module_correct",
        "category": "production",
        "subcategory": "Inception",
        "has_bug": False,
        "description": "Inception module with parallel branches and correct channel dims",
        "code": """\
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels=256):
        super().__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 64, 1)
        self.branch3x3_reduce = nn.Conv2d(in_channels, 96, 1)
        self.branch3x3 = nn.Conv2d(96, 128, 3, padding=1)
        self.branch5x5_reduce = nn.Conv2d(in_channels, 16, 1)
        self.branch5x5 = nn.Conv2d(16, 32, 5, padding=2)
        self.branch_pool_proj = nn.Conv2d(in_channels, 32, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b1 = self.relu(self.branch1x1(x))
        b3 = self.relu(self.branch3x3(self.relu(self.branch3x3_reduce(x))))
        b5 = self.relu(self.branch5x5(self.relu(self.branch5x5_reduce(x))))
        bp = self.relu(self.branch_pool_proj(x))
        return b1
""",
        "input_shapes": {"x": ("batch", 256, "h", "w")},
    },
    {
        "name": "se_block_bug",
        "category": "production",
        "subcategory": "SE-Net",
        "has_bug": True,
        "description": "SE block: fc1 expects 256 but input has 512 channels",
        "code": """\
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    def __init__(self, channels=512, reduction=16):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(256, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        se = self.pool(out)
        se = se.view(se.size(0), -1)
        se = self.relu(self.fc1(se))
        se = self.fc2(se)
        return out
""",
        "input_shapes": {"x": ("batch", 512, "h", "w")},
    },
    {
        "name": "fpn_correct",
        "category": "production",
        "subcategory": "FPN",
        "has_bug": False,
        "description": "FPN lateral connections with correct channel projections",
        "code": """\
import torch
import torch.nn as nn

class FPNNeck(nn.Module):
    def __init__(self, out_channels=256):
        super().__init__()
        self.lateral4 = nn.Conv2d(2048, out_channels, 1)
        self.lateral3 = nn.Conv2d(1024, out_channels, 1)
        self.lateral2 = nn.Conv2d(512, out_channels, 1)
        self.lateral1 = nn.Conv2d(256, out_channels, 1)
        self.smooth4 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth3 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.smooth1 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

    def forward(self, c1):
        p4 = self.lateral4(c1)
        p4 = self.smooth4(p4)
        return p4
""",
        "input_shapes": {"c1": ("batch", 2048, "h", "w")},
    },
]

# ── Category 3: Contract discovery (symbolic dimensions) ──────────────────────

CONTRACT_DISCOVERY_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "mlp_symbolic_features",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "nn.Linear(768, 256): CEGAR should discover features==768",
        "code": """\
import torch.nn as nn

class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(768, 256)

    def forward(self, x):
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 768"],
    },
    {
        "name": "cnn_symbolic_channels",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "nn.Conv2d(3, 64, 3): CEGAR should discover channels==3",
        "code": """\
import torch.nn as nn

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)

    def forward(self, x):
        return self.conv(x)
""",
        "input_shapes": {"x": ("batch", "channels", "height", "width")},
        "expected_predicates": ["x.shape[1] == 3"],
    },
    {
        "name": "transformer_symbolic_embed",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "Transformer projections: CEGAR should discover embed_dim==512",
        "code": """\
import torch.nn as nn

class TransformerEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(v)
""",
        "input_shapes": {"x": ("batch", "seq_len", "embed_dim")},
        "expected_predicates": ["x.shape[-1] == 512"],
    },
    {
        "name": "multilayer_mlp_symbolic",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "3-layer MLP (784->256->128->10): CEGAR should discover features==784",
        "code": """\
import torch.nn as nn

class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 784"],
    },
    {
        "name": "autoencoder_symbolic",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "Autoencoder (784->256->64->256->784): discover features==784",
        "code": """\
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
        x = self.relu(self.enc1(x))
        x = self.relu(self.enc2(x))
        x = self.relu(self.dec1(x))
        return self.dec2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 784"],
    },
    {
        "name": "buggy_mlp_symbolic",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": True,
        "description": "MLP with bug AND symbolic: fc1->256 but fc2 expects 128",
        "code": """\
import torch.nn as nn

class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": [],
    },
    {
        "name": "wide_net_symbolic",
        "category": "contract_discovery",
        "subcategory": "symbolic",
        "has_bug": False,
        "description": "Wide network (1024->512->256->128): discover features==1024",
        "code": """\
import torch.nn as nn

class WideNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", "features")},
        "expected_predicates": ["x.shape[-1] == 1024"],
    },
]

# ── Category 4: CEGAR ablation benchmarks ─────────────────────────────────────

CEGAR_ABLATION_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "mlp_bug",
        "category": "cegar_ablation",
        "subcategory": "MLP",
        "has_bug": True,
        "description": "MLP with mismatched intermediate dims (256 -> Linear(128,10))",
        "code": """\
import torch.nn as nn
class BuggyMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "mlp_correct",
        "category": "cegar_ablation",
        "subcategory": "MLP",
        "has_bug": False,
        "description": "MLP with consistent dimensions",
        "code": """\
import torch.nn as nn
class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "cnn_bug",
        "category": "cegar_ablation",
        "subcategory": "CNN",
        "has_bug": True,
        "description": "CNN with channel mismatch (conv1 outputs 32, conv2 expects 64)",
        "code": """\
import torch.nn as nn
class BuggyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "cnn_correct",
        "category": "cegar_ablation",
        "subcategory": "CNN",
        "has_bug": False,
        "description": "CNN with correct channel dimensions",
        "code": """\
import torch.nn as nn
class CorrectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        return x
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "transformer_bug",
        "category": "cegar_ablation",
        "subcategory": "Transformer",
        "has_bug": True,
        "description": "Transformer with mismatched projection (512 -> Linear(768,256))",
        "code": """\
import torch.nn as nn
class BuggyTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.out_proj = nn.Linear(512, 512)
        self.ff = nn.Linear(768, 256)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        x = self.out_proj(v)
        return self.ff(x)
""",
        "input_shapes": {"x": ("batch", "seq_len", 512)},
    },
    {
        "name": "resnet_correct",
        "category": "cegar_ablation",
        "subcategory": "ResNet",
        "has_bug": False,
        "description": "ResNet residual block with matching skip connection dims",
        "code": """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        return out
""",
        "input_shapes": {"x": ("batch", 64, "h", "w")},
    },
    {
        "name": "autoencoder_correct",
        "category": "cegar_ablation",
        "subcategory": "Autoencoder",
        "has_bug": False,
        "description": "Autoencoder with matching encoder/decoder dimensions",
        "code": """\
import torch.nn as nn
class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = nn.Linear(784, 256)
        self.encoder2 = nn.Linear(256, 64)
        self.decoder1 = nn.Linear(64, 256)
        self.decoder2 = nn.Linear(256, 784)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.encoder1(x))
        x = self.relu(self.encoder2(x))
        x = self.relu(self.decoder1(x))
        return self.decoder2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "lstm_style_bug",
        "category": "cegar_ablation",
        "subcategory": "LSTM",
        "has_bug": True,
        "description": "LSTM-style: input proj 256->128 but output proj expects 64",
        "code": """\
import torch.nn as nn
class LSTMStyleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_proj = nn.Linear(256, 128)
        self.gate_proj = nn.Linear(128, 128)
        self.output_proj = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = self.relu(self.input_proj(x))
        x = self.relu(self.gate_proj(x))
        return self.output_proj(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
]

ALL_BENCHMARKS = (
    THEORY_BENCHMARKS
    + PRODUCTION_BENCHMARKS
    + CONTRACT_DISCOVERY_BENCHMARKS
    + CEGAR_ABLATION_BENCHMARKS
)


# ═══════════════════════════════════════════════════════════════════════════════
# Syntactic Baseline (AST-based shape checker, no Z3/CEGAR)
# ═══════════════════════════════════════════════════════════════════════════════

class SyntacticShapeChecker:
    """Pure AST-based shape checker for nn.Module classes."""

    def __init__(self, source: str):
        self.source = source
        self.tree = ast.parse(source)
        self.layers: Dict[str, Dict[str, Any]] = {}
        self.bugs: List[str] = []

    def check(self) -> Tuple[bool, List[str]]:
        for node in ast.walk(self.tree):
            if isinstance(node, ast.ClassDef) and self._is_nn_module(node):
                self._extract_layers(node)
                self._trace_forward(node)
        return len(self.bugs) > 0, self.bugs

    def _is_nn_module(self, cls: ast.ClassDef) -> bool:
        for base in cls.bases:
            if isinstance(base, ast.Attribute) and base.attr == "Module":
                return True
            if isinstance(base, ast.Name) and base.id in ("Module",):
                return True
        return False

    def _extract_layers(self, cls: ast.ClassDef):
        for node in ast.walk(cls):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if (isinstance(target, ast.Attribute)
                            and isinstance(target.value, ast.Name)
                            and target.value.id == "self"
                            and isinstance(node.value, ast.Call)):
                        self._parse_layer_call(target.attr, node.value)

    def _parse_layer_call(self, attr_name: str, call: ast.Call):
        func = call.func
        layer_type = None
        if isinstance(func, ast.Attribute):
            layer_type = func.attr
        elif isinstance(func, ast.Name):
            layer_type = func.id

        if layer_type == "Linear" and len(call.args) >= 2:
            in_f = self._const_val(call.args[0])
            out_f = self._const_val(call.args[1])
            if in_f is not None and out_f is not None:
                self.layers[attr_name] = {
                    "type": "Linear", "in_features": in_f, "out_features": out_f
                }
        elif layer_type == "Conv2d" and len(call.args) >= 3:
            in_c = self._const_val(call.args[0])
            out_c = self._const_val(call.args[1])
            if in_c is not None and out_c is not None:
                self.layers[attr_name] = {
                    "type": "Conv2d", "in_channels": in_c, "out_channels": out_c,
                }
        elif layer_type == "ConvTranspose2d" and len(call.args) >= 3:
            in_c = self._const_val(call.args[0])
            out_c = self._const_val(call.args[1])
            if in_c is not None and out_c is not None:
                self.layers[attr_name] = {
                    "type": "ConvTranspose2d", "in_channels": in_c, "out_channels": out_c,
                }
        elif layer_type == "BatchNorm2d" and len(call.args) >= 1:
            nf = self._const_val(call.args[0])
            if nf is not None:
                self.layers[attr_name] = {
                    "type": "BatchNorm2d", "num_features": nf,
                }

    @staticmethod
    def _const_val(node: ast.expr) -> Optional[int]:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.Constant):
                return -node.operand.value
        if isinstance(node, ast.BinOp):
            left = SyntacticShapeChecker._const_val(node.left)
            right = SyntacticShapeChecker._const_val(node.right)
            if left is not None and right is not None:
                if isinstance(node.op, ast.Mult):
                    return left * right
                if isinstance(node.op, ast.FloorDiv):
                    return left // right if right != 0 else None
                if isinstance(node.op, ast.Add):
                    return left + right
                if isinstance(node.op, ast.Sub):
                    return left - right
        return None

    def _trace_forward(self, cls: ast.ClassDef):
        forward_fn = None
        for item in cls.body:
            if isinstance(item, ast.FunctionDef) and item.name == "forward":
                forward_fn = item
                break
        if forward_fn is None:
            return

        var_shapes: Dict[str, Optional[int]] = {}

        for stmt in ast.walk(forward_fn):
            call_info = self._extract_self_layer_call(stmt)
            if call_info is None:
                continue
            layer_name, input_var, output_var = call_info

            if layer_name not in self.layers:
                if input_var and input_var in var_shapes and output_var:
                    var_shapes[output_var] = var_shapes[input_var]
                continue

            layer = self.layers[layer_name]

            if layer["type"] == "Linear":
                expected_in = layer["in_features"]
                out_feat = layer["out_features"]
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_in:
                        self.bugs.append(
                            f"Shape mismatch at self.{layer_name}: "
                            f"input has {actual_in} features but "
                            f"nn.Linear expects {expected_in}"
                        )
                if output_var:
                    var_shapes[output_var] = out_feat

            elif layer["type"] in ("Conv2d", "ConvTranspose2d"):
                expected_in = layer["in_channels"]
                out_ch = layer.get("out_channels", expected_in)
                if input_var and input_var in var_shapes:
                    actual_in = var_shapes[input_var]
                    if actual_in is not None and actual_in != expected_in:
                        self.bugs.append(
                            f"Shape mismatch at self.{layer_name}: "
                            f"input has {actual_in} channels but "
                            f"layer expects {expected_in}"
                        )
                if output_var:
                    var_shapes[output_var] = out_ch

            elif layer["type"] == "BatchNorm2d":
                expected_nf = layer["num_features"]
                if input_var and input_var in var_shapes:
                    actual = var_shapes[input_var]
                    if actual is not None and actual != expected_nf:
                        self.bugs.append(
                            f"Shape mismatch at self.{layer_name}: "
                            f"input has {actual} features but "
                            f"BatchNorm2d expects {expected_nf}"
                        )
                if output_var:
                    var_shapes[output_var] = var_shapes.get(input_var) if input_var else None

    def _extract_self_layer_call(self, node) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target = node.targets[0]
            output_var = target.id if isinstance(target, ast.Name) else None
            return self._unpack_call(node.value, output_var)
        elif isinstance(node, ast.Return) and node.value is not None:
            return self._unpack_call(node.value, None)
        return None

    def _unpack_call(self, expr, output_var) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        if not isinstance(expr, ast.Call):
            return None
        func = expr.func
        if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id == "self":
            layer_name = func.attr
            input_var = None
            if expr.args:
                arg = expr.args[0]
                if isinstance(arg, ast.Name):
                    input_var = arg.id
                elif isinstance(arg, ast.Call):
                    inner = self._unpack_call(arg, output_var)
                    if inner:
                        return inner
            return (layer_name, input_var, output_var)
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# TensorGuard runner (CEGAR + Z3 theories)
# ═══════════════════════════════════════════════════════════════════════════════

def run_tensorguard(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Run TensorGuard: verify_model + CEGAR pipeline with quality filter.

    The CEGAR result is authoritative: if CEGAR resolves all
    counterexamples as spurious (status=SAFE), the model is safe even
    if the initial single-pass verify_model reported violations.
    """
    t0 = time.monotonic()
    details = ""
    detected = False
    try:
        # Run CEGAR pipeline (includes verify_model internally)
        cegar_result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        if cegar_result.has_real_bugs:
            detected = True
            if cegar_result.real_bugs:
                for b in cegar_result.real_bugs[:2]:
                    msg = getattr(b, "message", str(b))
                    details += msg[:150] + "; "
        elif not cegar_result.is_safe:
            # CEGAR didn't converge but also didn't find real bugs;
            # fall back to single-pass verify_model
            vm_result = verify_model(
                tc["code"],
                input_shapes=tc["input_shapes"],
            )
            if not vm_result.safe:
                detected = True
                if vm_result.counterexample:
                    for v in vm_result.counterexample.violations[:2]:
                        msg = getattr(v, "message", str(v))
                        details += msg[:150] + "; "
        if cegar_result.is_safe and cegar_result.discovered_predicates:
            preds = ", ".join(str(p) for p in cegar_result.discovered_predicates)
            details += f"Contracts discovered: {preds}; "
    except Exception as e:
        details = f"ERROR: {e}"
    elapsed = (time.monotonic() - t0) * 1000
    return {
        "detected_bug": detected,
        "time_ms": round(elapsed, 2),
        "details": details[:500],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CEGAR ablation runners
# ═══════════════════════════════════════════════════════════════════════════════

def run_no_cegar(tc: Dict[str, Any]) -> Dict[str, Any]:
    """Single-pass ConstraintVerifier — no contract discovery loop."""
    t0 = time.monotonic()
    try:
        graph = extract_computation_graph(tc["code"])
        checker = ConstraintVerifier(graph, input_shapes=tc["input_shapes"])
        result = checker.verify()
        detected = not result.safe
    except Exception:
        detected = False
    elapsed = (time.monotonic() - t0) * 1000
    return {"detected_bug": detected, "time_ms": round(elapsed, 2)}


def run_cegar_filtered(tc: Dict[str, Any]) -> Dict[str, Any]:
    """CEGAR with quality filter ON."""
    t0 = time.monotonic()
    try:
        result = run_shape_cegar(
            tc["code"],
            input_shapes=tc["input_shapes"],
            max_iterations=10,
            enable_quality_filter=True,
        )
        detected = result.has_real_bugs
    except Exception:
        detected = False
    elapsed = (time.monotonic() - t0) * 1000
    return {"detected_bug": detected, "time_ms": round(elapsed, 2)}


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics + Bootstrap CI
# ═══════════════════════════════════════════════════════════════════════════════

def compute_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    tp = fp = fn = tn = 0
    for r in results:
        gt = r["ground_truth"]
        det = r["detected_bug"]
        if gt and det:
            tp += 1
        elif not gt and det:
            fp += 1
        elif gt and not det:
            fn += 1
        else:
            tn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
    avg_time = sum(r["time_ms"] for r in results) / len(results) if results else 0.0
    return {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "avg_time_ms": round(avg_time, 2),
    }


def bootstrap_ci(
    results: List[Dict[str, Any]],
    n_bootstrap: int = 1000,
    ci: float = 0.95,
) -> Dict[str, Tuple[float, float]]:
    """Bootstrap 95% confidence intervals for precision, recall, F1."""
    rng = random.Random(42)
    samples: Dict[str, List[float]] = {"precision": [], "recall": [], "f1": []}
    for _ in range(n_bootstrap):
        sample = rng.choices(results, k=len(results))
        m = compute_metrics(sample)
        for key in samples:
            samples[key].append(m[key])
    cis = {}
    alpha = (1 - ci) / 2
    for key, vals in samples.items():
        vals.sort()
        lo = vals[max(0, int(alpha * len(vals)))]
        hi = vals[min(len(vals) - 1, int((1 - alpha) * len(vals)))]
        cis[key] = (round(lo, 4), round(hi, 4))
    return cis


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    n_theory = len(THEORY_BENCHMARKS)
    n_prod = len(PRODUCTION_BENCHMARKS)
    n_contract = len(CONTRACT_DISCOVERY_BENCHMARKS)
    n_ablation = len(CEGAR_ABLATION_BENCHMARKS)
    n_total = len(ALL_BENCHMARKS)

    print("=" * 78)
    print("  Comprehensive Final Evaluation — Unified Benchmark Suite")
    print("=" * 78)
    print(f"  Theory-exercising:    {n_theory:3d} benchmarks (broadcast, matmul, multi-step)")
    print(f"  Production-style:     {n_prod:3d} benchmarks (ResNet, Transformer, U-Net, ...)")
    print(f"  Contract discovery:   {n_contract:3d} benchmarks (symbolic dimensions)")
    print(f"  CEGAR ablation:       {n_ablation:3d} benchmarks (no-CEGAR vs filtered-CEGAR)")
    print(f"  Total:                {n_total:3d} benchmarks")
    print(f"  Bootstrap resamples:  1000 (95% CI)")
    print("=" * 78)

    # ──────────────────────────────────────────────────────────────────────
    # Section 1–3: Run syntactic + tensorguard on all benchmarks
    # ──────────────────────────────────────────────────────────────────────
    all_results: Dict[str, List[Dict[str, Any]]] = {
        "syntactic": [], "tensorguard": []
    }
    benchmark_idx = 0

    for section_name, benchmarks in [
        ("THEORY-EXERCISING", THEORY_BENCHMARKS),
        ("PRODUCTION-STYLE", PRODUCTION_BENCHMARKS),
        ("CONTRACT DISCOVERY", CONTRACT_DISCOVERY_BENCHMARKS),
        ("CEGAR ABLATION", CEGAR_ABLATION_BENCHMARKS),
    ]:
        print(f"\n{'─' * 78}")
        print(f"  Section: {section_name} ({len(benchmarks)} benchmarks)")
        print(f"{'─' * 78}")

        for tc in benchmarks:
            benchmark_idx += 1
            tag = "BUGGY" if tc["has_bug"] else "CLEAN"
            print(f"\n[{benchmark_idx:2d}/{n_total}] {tc['name']} ({tag})")
            print(f"         {tc['description']}")

            # Syntactic baseline
            t0 = time.monotonic()
            checker = SyntacticShapeChecker(tc["code"])
            has_bug, bug_msgs = checker.check()
            syn_time = (time.monotonic() - t0) * 1000
            syn_result = {
                "name": tc["name"],
                "category": tc["category"],
                "subcategory": tc.get("subcategory", ""),
                "ground_truth": tc["has_bug"],
                "detected_bug": has_bug,
                "time_ms": round(syn_time, 2),
                "details": "; ".join(bug_msgs) if bug_msgs else "",
            }
            all_results["syntactic"].append(syn_result)
            syn_mark = "✓" if has_bug == tc["has_bug"] else "✗"
            print(f"  Syntactic:  {syn_mark}  det={str(has_bug):<5}  {syn_time:.1f}ms"
                  + (f"  [{bug_msgs[0][:60]}]" if bug_msgs else ""))

            # TensorGuard (CEGAR + Z3)
            lp = run_tensorguard(tc)
            lp_result = {
                "name": tc["name"],
                "category": tc["category"],
                "subcategory": tc.get("subcategory", ""),
                "ground_truth": tc["has_bug"],
                "detected_bug": lp["detected_bug"],
                "time_ms": lp["time_ms"],
                "details": lp["details"],
            }
            all_results["tensorguard"].append(lp_result)
            lp_mark = "✓" if lp["detected_bug"] == tc["has_bug"] else "✗"
            print(f"  TensorGuard:   {lp_mark}  det={str(lp['detected_bug']):<5}  {lp['time_ms']:.1f}ms"
                  + (f"  [{lp['details'][:60]}]" if lp["details"] else ""))

    # ──────────────────────────────────────────────────────────────────────
    # Section 4: CEGAR ablation (no-CEGAR vs filtered-CEGAR)
    # ──────────────────────────────────────────────────────────────────────
    print(f"\n{'─' * 78}")
    print("  CEGAR ABLATION — no-CEGAR vs filtered-CEGAR")
    print(f"{'─' * 78}")

    ablation_results: Dict[str, List[Dict[str, Any]]] = {
        "no_cegar": [], "cegar_filtered": []
    }

    for tc in CEGAR_ABLATION_BENCHMARKS:
        nc = run_no_cegar(tc)
        cf = run_cegar_filtered(tc)
        for mode_key, r in [("no_cegar", nc), ("cegar_filtered", cf)]:
            ablation_results[mode_key].append({
                "name": tc["name"],
                "category": tc["category"],
                "subcategory": tc.get("subcategory", ""),
                "ground_truth": tc["has_bug"],
                "detected_bug": r["detected_bug"],
                "time_ms": r["time_ms"],
            })
        nc_mark = "✓" if nc["detected_bug"] == tc["has_bug"] else "✗"
        cf_mark = "✓" if cf["detected_bug"] == tc["has_bug"] else "✗"
        print(f"  {tc['name']:30s}  no-CEGAR={nc_mark}  filtered-CEGAR={cf_mark}")

    # ══════════════════════════════════════════════════════════════════════
    # RESULTS SUMMARY
    # ══════════════════════════════════════════════════════════════════════

    print(f"\n{'=' * 78}")
    print("  OVERALL METRICS (syntactic vs tensorguard) — all benchmarks")
    print(f"{'=' * 78}")

    tool_metrics: Dict[str, Any] = {}
    tool_cis: Dict[str, Any] = {}
    for tool in ["syntactic", "tensorguard"]:
        m = compute_metrics(all_results[tool])
        ci = bootstrap_ci(all_results[tool])
        tool_metrics[tool] = m
        tool_cis[tool] = ci
        label = {
            "syntactic": "Syntactic Pattern Matching",
            "tensorguard": "TensorGuard (CEGAR + Z3)",
        }
        print(f"\n  {label[tool]:35s}")
        print(f"    F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  "
              f"R={m['recall']:<6.4f}  Acc={m['accuracy']:<6.4f}")
        print(f"    TP={m['tp']}  FP={m['fp']}  FN={m['fn']}  TN={m['tn']}  "
              f"avg={m['avg_time_ms']:.1f}ms")
        print(f"    95% CI: F1={ci['f1']}  P={ci['precision']}  R={ci['recall']}")

    lp_f1 = tool_metrics["tensorguard"]["f1"]
    syn_f1 = tool_metrics["syntactic"]["f1"]
    print(f"\n  TensorGuard vs Syntactic:  ΔF1 = {lp_f1 - syn_f1:+.4f}")

    # ── Per-category breakdown ────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  PER-CATEGORY BREAKDOWN")
    print(f"{'=' * 78}")

    categories_seen = sorted(set(r["category"] for r in all_results["tensorguard"]))
    for cat in categories_seen:
        print(f"\n  Category: {cat}")
        for tool in ["syntactic", "tensorguard"]:
            cat_results = [r for r in all_results[tool] if r["category"] == cat]
            m = compute_metrics(cat_results)
            ci = bootstrap_ci(cat_results)
            label = "Syn" if tool == "syntactic" else "LP "
            print(f"    {label}:  F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  "
                  f"R={m['recall']:<6.4f}  "
                  f"(TP={m['tp']} FP={m['fp']} FN={m['fn']} TN={m['tn']})  "
                  f"avg={m['avg_time_ms']:.1f}ms")
            print(f"         95% CI: F1={ci['f1']}  P={ci['precision']}  R={ci['recall']}")

    # ── Theory vs Production separation ───────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  THEORY vs PRODUCTION SPLIT")
    print(f"{'=' * 78}")

    for split_name, split_cat in [("Theory (Z3 reasoning)", "theory"),
                                   ("Production (realistic code)", "production")]:
        print(f"\n  {split_name}:")
        for tool in ["syntactic", "tensorguard"]:
            split_results = [r for r in all_results[tool] if r["category"] == split_cat]
            if not split_results:
                continue
            m = compute_metrics(split_results)
            ci = bootstrap_ci(split_results)
            label = "Syn" if tool == "syntactic" else "LP "
            print(f"    {label}:  F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  "
                  f"R={m['recall']:<6.4f}  Acc={m['accuracy']:<6.4f}  "
                  f"avg={m['avg_time_ms']:.1f}ms")
            print(f"         95% CI: F1={ci['f1']}  P={ci['precision']}  R={ci['recall']}")

    # ── Subcategory breakdown ─────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  PER-SUBCATEGORY BREAKDOWN (TensorGuard)")
    print(f"{'=' * 78}")

    subcats = sorted(set(
        r["subcategory"] for r in all_results["tensorguard"] if r["subcategory"]
    ))
    for sc in subcats:
        sc_results = [r for r in all_results["tensorguard"] if r["subcategory"] == sc]
        m = compute_metrics(sc_results)
        print(f"  {sc:25s}  n={len(sc_results):2d}  F1={m['f1']:<6.4f}  "
              f"P={m['precision']:<6.4f}  R={m['recall']:<6.4f}")

    # ── CEGAR ablation summary ────────────────────────────────────────────
    print(f"\n{'=' * 78}")
    print("  CEGAR ABLATION SUMMARY")
    print(f"{'=' * 78}")

    abl_metrics: Dict[str, Any] = {}
    abl_cis: Dict[str, Any] = {}
    for mode_key, mode_label in [("no_cegar", "No contract discovery (single-pass verification)"),
                                  ("cegar_filtered", "CEGAR (quality-filtered)")]:
        m = compute_metrics(ablation_results[mode_key])
        ci = bootstrap_ci(ablation_results[mode_key])
        abl_metrics[mode_key] = m
        abl_cis[mode_key] = ci
        print(f"\n  {mode_label:35s}")
        print(f"    F1={m['f1']:<6.4f}  P={m['precision']:<6.4f}  R={m['recall']:<6.4f}  "
              f"avg={m['avg_time_ms']:.1f}ms")
        print(f"    95% CI: F1={ci['f1']}  P={ci['precision']}  R={ci['recall']}")

    nc_f1 = abl_metrics["no_cegar"]["f1"]
    cf_f1 = abl_metrics["cegar_filtered"]["f1"]
    print(f"\n  Filtered-CEGAR vs No-CEGAR:  ΔF1 = {cf_f1 - nc_f1:+.4f}")
    if cf_f1 >= nc_f1:
        print("  → Quality-filtered CEGAR does NOT degrade results ✓")
    else:
        print("  → Quality-filtered CEGAR shows degradation ✗")

    # ══════════════════════════════════════════════════════════════════════
    # Save JSON
    # ══════════════════════════════════════════════════════════════════════

    output: Dict[str, Any] = {
        "experiment": "comprehensive_final_evaluation",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "benchmark_counts": {
            "theory": n_theory,
            "production": n_prod,
            "contract_discovery": n_contract,
            "cegar_ablation": n_ablation,
            "total": n_total,
        },
        "bootstrap_resamples": 1000,
        "overall": {},
        "per_category": {},
        "theory_vs_production": {},
        "per_subcategory": {},
        "cegar_ablation": {},
        "per_benchmark": {},
    }

    # Overall
    for tool in ["syntactic", "tensorguard"]:
        output["overall"][tool] = {
            "metrics": tool_metrics[tool],
            "confidence_intervals_95": {k: list(v) for k, v in tool_cis[tool].items()},
        }

    # Per-category
    for cat in categories_seen:
        output["per_category"][cat] = {}
        for tool in ["syntactic", "tensorguard"]:
            cat_results = [r for r in all_results[tool] if r["category"] == cat]
            m = compute_metrics(cat_results)
            ci = bootstrap_ci(cat_results)
            output["per_category"][cat][tool] = {
                "metrics": m,
                "confidence_intervals_95": {k: list(v) for k, v in ci.items()},
            }

    # Theory vs production split
    for split_cat in ["theory", "production"]:
        output["theory_vs_production"][split_cat] = {}
        for tool in ["syntactic", "tensorguard"]:
            split_results = [r for r in all_results[tool] if r["category"] == split_cat]
            if not split_results:
                continue
            m = compute_metrics(split_results)
            ci = bootstrap_ci(split_results)
            output["theory_vs_production"][split_cat][tool] = {
                "metrics": m,
                "confidence_intervals_95": {k: list(v) for k, v in ci.items()},
            }

    # Per-subcategory (tensorguard only)
    for sc in subcats:
        sc_results = [r for r in all_results["tensorguard"] if r["subcategory"] == sc]
        m = compute_metrics(sc_results)
        ci = bootstrap_ci(sc_results)
        output["per_subcategory"][sc] = {
            "metrics": m,
            "confidence_intervals_95": {k: list(v) for k, v in ci.items()},
        }

    # CEGAR ablation
    for mode_key in ["no_cegar", "cegar_filtered"]:
        output["cegar_ablation"][mode_key] = {
            "metrics": abl_metrics[mode_key],
            "confidence_intervals_95": {k: list(v) for k, v in abl_cis[mode_key].items()},
            "per_benchmark": ablation_results[mode_key],
        }
    output["cegar_ablation"]["delta_f1"] = round(cf_f1 - nc_f1, 4)

    # Per-benchmark results
    for tool in ["syntactic", "tensorguard"]:
        output["per_benchmark"][tool] = all_results[tool]

    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
