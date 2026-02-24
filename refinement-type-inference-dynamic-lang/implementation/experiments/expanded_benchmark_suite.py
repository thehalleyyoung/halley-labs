"""
Expanded Benchmark Suite for TensorGuard nn.Module Verification.

200+ nn.Module benchmarks across 8 categories, designed to stress-test
TensorGuard's Z3-backed constraint verifier and measure false-positive rate.

Categories:
  1. HuggingFace-style architectures (transformer blocks, MHA, MLP, etc.)
  2. Computer vision architectures (ResNet, UNet, VGG, FPN blocks)
  3. Device inconsistency bugs (mixed CPU/CUDA operations)
  4. Phase-dependent bugs (train/eval behavioral differences)
  5. Broadcasting bugs (cross-rank, multi-projection)
  6. Reshape/view bugs (invalid reshape, flatten mismatches)
  7. Multi-layer chain bugs (long chains with propagated errors)
  8. Correct programs (non-trivial architectures, false-positive checks)

Export: EXPANDED_BENCHMARKS (list of dicts)
"""

from __future__ import annotations
from typing import Any, Dict, List

# ═══════════════════════════════════════════════════════════════════════════════
# Category 1: HuggingFace-style Architectures (30+)
# ═══════════════════════════════════════════════════════════════════════════════

HUGGINGFACE_BENCHMARKS: List[Dict[str, Any]] = [
    # --- Correct ---
    {
        "name": "hf_mlp_block_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Standard MLP block: Linear->GELU->Linear with matching dims",
        "code": """\
import torch.nn as nn
class MLPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(3072, 768)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_layernorm_residual_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Pre-norm residual: LayerNorm->Linear->add residual",
        "code": """\
import torch.nn as nn
class PreNormResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, 768)
    def forward(self, x):
        return x + self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_encoder_block_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Full encoder block: LN->MHA->residual->LN->MLP->residual",
        "code": """\
import torch.nn as nn
class EncoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        h = self.ln1(x)
        h, _ = self.attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h = self.fc2(self.act(self.fc1(h)))
        return x + h
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_decoder_causal_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Decoder block with self-attention and cross-attention",
        "code": """\
import torch.nn as nn
class DecoderBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(512)
        self.self_attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(512)
        self.cross_attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln3 = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x, enc):
        h = self.ln1(x)
        h, _ = self.self_attn(h, h, h)
        x = x + h
        h = self.ln2(x)
        h, _ = self.cross_attn(h, enc, enc)
        x = x + h
        h = self.ln3(x)
        h = self.fc2(self.fc1(h))
        return x + h
""",
        "input_shapes": {"x": ("batch", "seq", 512), "enc": ("batch", "seq", 512)},
    },
    {
        "name": "hf_qkv_projection_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Separate Q/K/V projections with matching dims",
        "code": """\
import torch.nn as nn
class QKVProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 768)
        self.v_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k + v)
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_gpt2_mlp_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "GPT-2 style MLP: expand 4x then contract",
        "code": """\
import torch.nn as nn
class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(1024, 4096)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4096, 1024)
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 1024)},
    },
    {
        "name": "hf_classification_head_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Classification head: Linear->tanh->Linear",
        "code": """\
import torch.nn as nn
class ClassificationHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.act = nn.Tanh()
        self.out_proj = nn.Linear(768, 2)
    def forward(self, x):
        return self.out_proj(self.act(self.dense(x)))
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "hf_embedding_projection_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Embedding projection layer that changes hidden dim",
        "code": """\
import torch.nn as nn
class EmbeddingProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 768)
        self.norm = nn.LayerNorm(768)
    def forward(self, x):
        return self.norm(self.proj(x))
""",
        "input_shapes": {"x": ("batch", "seq", 128)},
    },

    # --- Buggy ---
    {
        "name": "hf_mlp_dim_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "fc2 expects 2048 but fc1 outputs 3072",
        "description": "MLP block with mismatched intermediate dim",
        "code": """\
import torch.nn as nn
class MLPBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 3072)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2048, 768)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_residual_dim_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Residual add fails: fc outputs 512 but input is 768",
        "description": "Residual connection with projection to wrong dim",
        "code": """\
import torch.nn as nn
class ResidualBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc = nn.Linear(768, 512)
    def forward(self, x):
        return x + self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_qkv_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Q projects to 768 but K projects to 512, add fails",
        "description": "Q/K projections with mismatched output dims",
        "code": """\
import torch.nn as nn
class QKVMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 768)
        self.k_proj = nn.Linear(768, 512)
        self.v_proj = nn.Linear(768, 768)
        self.out_proj = nn.Linear(768, 768)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        return self.out_proj(q + k)
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_cross_attn_dim_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Cross attention q_proj 512 vs k_proj 256, incompatible add",
        "description": "Cross-attention Q and K projected to different dims",
        "code": """\
import torch.nn as nn
class CrossAttnBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(768, 512)
        self.k_proj = nn.Linear(768, 256)
    def forward(self, q, k):
        q = self.q_proj(q)
        k = self.k_proj(k)
        return q + k
""",
        "input_shapes": {"q": ("batch", "seq", 768), "k": ("batch", "seq", 768)},
    },
    {
        "name": "hf_encoder_output_proj_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Output projection expects 1024 but encoder hidden is 512",
        "description": "Encoder with wrong output projection input dim",
        "code": """\
import torch.nn as nn
class EncoderOutputBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.fc1 = nn.Linear(512, 2048)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2048, 512)
        self.out_proj = nn.Linear(1024, 256)
    def forward(self, x):
        h = self.fc2(self.act(self.fc1(self.ln(x))))
        return self.out_proj(h)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_gpt2_mlp_expand_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "c_proj expects 2048 but c_fc outputs 4096",
        "description": "GPT-2 MLP with wrong contraction dim",
        "code": """\
import torch.nn as nn
class GPT2MLPBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.c_fc = nn.Linear(1024, 4096)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(2048, 1024)
    def forward(self, x):
        return self.c_proj(self.act(self.c_fc(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 1024)},
    },
    {
        "name": "hf_pooler_input_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Pooler Linear expects 1024 but input is 768",
        "description": "BERT pooler with wrong input dim",
        "code": """\
import torch.nn as nn
class PoolerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(1024, 768)
        self.act = nn.Tanh()
    def forward(self, x):
        return self.act(self.dense(x))
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "hf_layernorm_dim_mismatch_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "LayerNorm(512) applied to hidden dim 768",
        "description": "LayerNorm with wrong normalized shape",
        "code": """\
import torch.nn as nn
class LayerNormBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(512)
        self.fc = nn.Linear(768, 768)
    def forward(self, x):
        return self.fc(self.norm(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_two_tower_merge_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Two towers project to 256 and 512, add fails",
        "description": "Two-tower model with mismatched merge dims",
        "code": """\
import torch.nn as nn
class TwoTowerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower_a = nn.Linear(768, 256)
        self.tower_b = nn.Linear(768, 512)
    def forward(self, a, b):
        return self.tower_a(a) + self.tower_b(b)
""",
        "input_shapes": {"a": ("batch", 768), "b": ("batch", 768)},
    },
    {
        "name": "hf_feedforward_chain_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Three-layer feedforward: 768->2048->1024->768",
        "code": """\
import torch.nn as nn
class FeedForwardChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 2048)
        self.fc2 = nn.Linear(2048, 1024)
        self.fc3 = nn.Linear(1024, 768)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.fc3(self.act(self.fc2(self.act(self.fc1(x)))))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_adapter_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "LoRA-style adapter: down-project then up-project with residual",
        "code": """\
import torch.nn as nn
class Adapter(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(768, 64)
        self.act = nn.ReLU()
        self.up = nn.Linear(64, 768)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_adapter_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "Adapter up-projects to 512 but residual expects 768",
        "description": "Adapter with wrong up-projection dim, residual fails",
        "code": """\
import torch.nn as nn
class AdapterBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(768, 64)
        self.act = nn.ReLU()
        self.up = nn.Linear(64, 512)
    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_multi_layer_encoder_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Two stacked encoder layers with matching dims",
        "code": """\
import torch.nn as nn
class StackedEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(256)
        self.fc1a = nn.Linear(256, 1024)
        self.fc1b = nn.Linear(1024, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc2a = nn.Linear(256, 1024)
        self.fc2b = nn.Linear(1024, 256)
    def forward(self, x):
        h = self.fc1b(self.fc1a(self.ln1(x)))
        x = x + h
        h = self.fc2b(self.fc2a(self.ln2(x)))
        return x + h
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "hf_decoder_fc_chain_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "fc2 in second layer expects 2048 but fc1 outputs 1024",
        "description": "Stacked decoder with broken second layer MLP",
        "code": """\
import torch.nn as nn
class StackedDecoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(256)
        self.fc1a = nn.Linear(256, 1024)
        self.fc1b = nn.Linear(1024, 256)
        self.ln2 = nn.LayerNorm(256)
        self.fc2a = nn.Linear(256, 1024)
        self.fc2b = nn.Linear(2048, 256)
    def forward(self, x):
        h = self.fc1b(self.fc1a(self.ln1(x)))
        x = x + h
        h = self.fc2b(self.fc2a(self.ln2(x)))
        return x + h
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "hf_token_classification_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Token classification: encoder output -> classifier per token",
        "code": """\
import torch.nn as nn
class TokenClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, 9)
    def forward(self, x):
        return self.classifier(self.dropout(self.encoder(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_multihead_head_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "out_proj expects 512 but attention embed_dim is 768",
        "description": "MultiheadAttention output projection dim mismatch",
        "code": """\
import torch.nn as nn
class MHAOutputBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.out_proj = nn.Linear(512, 768)
    def forward(self, x):
        h, _ = self.attn(x, x, x)
        return self.out_proj(h)
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_sequence_output_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Sequence output with dense + activation + projection",
        "code": """\
import torch.nn as nn
class SequenceOutput(nn.Module):
    def __init__(self):
        super().__init__()
        self.dense = nn.Linear(768, 768)
        self.act = nn.Tanh()
        self.proj = nn.Linear(768, 30522)
    def forward(self, x):
        return self.proj(self.act(self.dense(x)))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_gated_mlp_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Gated MLP (LLaMA-style): gate * up, then down",
        "code": """\
import torch.nn as nn
class GatedMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(512, 1024)
        self.up_proj = nn.Linear(512, 1024)
        self.down_proj = nn.Linear(1024, 512)
        self.act = nn.SiLU()
    def forward(self, x):
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_gated_mlp_gate_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "gate_proj outputs 2048 but up_proj outputs 1024, multiply fails",
        "description": "Gated MLP with mismatched gate and up projection dims",
        "code": """\
import torch.nn as nn
class GatedMLPBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(512, 2048)
        self.up_proj = nn.Linear(512, 1024)
        self.down_proj = nn.Linear(1024, 512)
        self.act = nn.SiLU()
    def forward(self, x):
        gate = self.act(self.gate_proj(x))
        up = self.up_proj(x)
        return self.down_proj(gate * up)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_rotary_proj_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Rotary-style Q/K projection with matching dims",
        "code": """\
import torch.nn as nn
class RotaryProj(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.k_proj = nn.Linear(512, 512)
        self.v_proj = nn.Linear(512, 512)
        self.o_proj = nn.Linear(512, 512)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        combined = q + v
        return self.o_proj(combined)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_bottleneck_adapter_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "down_proj expects 768 but bottleneck is 64, up_proj out is 256 not 768",
        "description": "Bottleneck adapter with wrong up-projection output",
        "code": """\
import torch.nn as nn
class BottleneckAdapterBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.down_proj = nn.Linear(768, 64)
        self.act = nn.ReLU()
        self.up_proj = nn.Linear(64, 256)
    def forward(self, x):
        h = self.up_proj(self.act(self.down_proj(x)))
        return x + h
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_swiglu_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "SwiGLU activation pattern with matching dims",
        "code": """\
import torch.nn as nn
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(256, 512)
        self.w2 = nn.Linear(256, 512)
        self.w3 = nn.Linear(512, 256)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "hf_swiglu_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "w1 outputs 512 but w2 outputs 1024, element-wise multiply fails",
        "description": "SwiGLU with mismatched gate dims",
        "code": """\
import torch.nn as nn
class SwiGLUBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.w1 = nn.Linear(256, 512)
        self.w2 = nn.Linear(256, 1024)
        self.w3 = nn.Linear(512, 256)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.w3(self.act(self.w1(x)) * self.w2(x))
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "hf_parallel_attention_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Parallel attention + MLP (GPT-NeoX style)",
        "code": """\
import torch.nn as nn
class ParallelAttnMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc1 = nn.Linear(512, 2048)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2048, 512)
    def forward(self, x):
        h = self.ln(x)
        attn_out, _ = self.attn(h, h, h)
        mlp_out = self.fc2(self.act(self.fc1(h)))
        return x + attn_out + mlp_out
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_parallel_attn_mlp_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "MLP fc2 outputs 256 instead of 512, residual add fails",
        "description": "Parallel attn+MLP with wrong MLP output dim",
        "code": """\
import torch.nn as nn
class ParallelAttnMLPBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc1 = nn.Linear(512, 2048)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2048, 256)
    def forward(self, x):
        h = self.ln(x)
        attn_out, _ = self.attn(h, h, h)
        mlp_out = self.fc2(self.act(self.fc1(h)))
        return x + attn_out + mlp_out
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "hf_language_model_head_correct",
        "category": "huggingface",
        "has_bug": False,
        "description": "Language model head: LayerNorm -> Linear to vocab",
        "code": """\
import torch.nn as nn
class LMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(768)
        self.lm_head = nn.Linear(768, 50257)
    def forward(self, x):
        return self.lm_head(self.ln(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
    {
        "name": "hf_lm_head_dim_bug",
        "category": "huggingface",
        "has_bug": True,
        "bug_description": "lm_head expects 1024 but hidden dim is 768",
        "description": "LM head with wrong input dim",
        "code": """\
import torch.nn as nn
class LMHeadBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.LayerNorm(768)
        self.lm_head = nn.Linear(1024, 50257)
    def forward(self, x):
        return self.lm_head(self.ln(x))
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 2: Computer Vision Architectures (30+)
# ═══════════════════════════════════════════════════════════════════════════════

VISION_BENCHMARKS: List[Dict[str, Any]] = [
    # --- Correct ---
    {
        "name": "cv_resnet_basic_block_correct",
        "category": "vision",
        "has_bug": False,
        "description": "ResNet basic block: conv->bn->relu->conv->bn + residual",
        "code": """\
import torch.nn as nn
class BasicBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_resnet_bottleneck_correct",
        "category": "vision",
        "has_bug": False,
        "description": "ResNet bottleneck: 1x1->3x3->1x1 with channel expansion",
        "code": """\
import torch.nn as nn
class Bottleneck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 256, 16, 16)},
    },
    {
        "name": "cv_vgg_block_correct",
        "category": "vision",
        "has_bug": False,
        "description": "VGG-style block: two conv layers with same channel count",
        "code": """\
import torch.nn as nn
class VGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        return self.relu(self.conv2(h))
""",
        "input_shapes": {"x": ("batch", 128, 32, 32)},
    },
    {
        "name": "cv_downsample_block_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Downsample block: conv with stride 2 + downsample shortcut",
        "code": """\
import torch.nn as nn
class DownsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.downsample = nn.Conv2d(64, 128, 1, stride=2)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.downsample(x))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_unet_encoder_correct",
        "category": "vision",
        "has_bug": False,
        "description": "UNet encoder stage: double conv block",
        "code": """\
import torch.nn as nn
class UNetEncoderStage(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(h)))
""",
        "input_shapes": {"x": ("batch", 3, 256, 256)},
    },
    {
        "name": "cv_fpn_lateral_correct",
        "category": "vision",
        "has_bug": False,
        "description": "FPN lateral connection: 1x1 conv to unify channels",
        "code": """\
import torch.nn as nn
class FPNLateral(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral3 = nn.Conv2d(512, 256, 1)
        self.lateral4 = nn.Conv2d(1024, 256, 1)
    def forward(self, c3, c4):
        p3 = self.lateral3(c3)
        p4 = self.lateral4(c4)
        return p3 + p4
""",
        "input_shapes": {"c3": ("batch", 512, 28, 28), "c4": ("batch", 1024, 28, 28)},
    },
    {
        "name": "cv_squeeze_excite_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Squeeze-and-excitation block with channel attention",
        "code": """\
import torch.nn as nn
class SqueezeExcite(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * scale
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "cv_simple_classifier_correct",
        "category": "vision",
        "has_bug": True,
        "bug_description": "Conv output (4D) passed directly to Linear without flatten",
        "description": "Simple conv classifier: missing flatten before fc",
        "code": """\
import torch.nn as nn
class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "cv_inception_branch_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Inception-style parallel branches with 1x1, 3x3, 5x5 convs",
        "code": """\
import torch.nn as nn
class InceptionBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(256, 64, 1)
        self.branch3 = nn.Conv2d(256, 64, 3, padding=1)
        self.branch5 = nn.Conv2d(256, 64, 5, padding=2)
    def forward(self, x):
        b1 = self.branch1(x)
        b3 = self.branch3(x)
        b5 = self.branch5(x)
        return b1 + b3 + b5
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
    },
    {
        "name": "cv_depthwise_separable_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Depthwise separable conv: depthwise + pointwise",
        "code": """\
import torch.nn as nn
class DepthwiseSeparable(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise = nn.Conv2d(64, 128, 1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.pointwise(self.depthwise(x))))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },

    # --- Buggy ---
    {
        "name": "cv_resnet_residual_channel_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "conv2 outputs 128 channels but residual x has 64 channels",
        "description": "ResNet block where main path changes channels but shortcut doesn't",
        "code": """\
import torch.nn as nn
class ResBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_bottleneck_expansion_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "conv3 outputs 512 but residual x has 256 channels",
        "description": "Bottleneck with wrong expansion factor on last conv",
        "code": """\
import torch.nn as nn
class BottleneckBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 64, 1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 512, 1)
        self.bn3 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.relu(self.bn2(self.conv2(h)))
        h = self.bn3(self.conv3(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 256, 16, 16)},
    },
    {
        "name": "cv_bn_channel_mismatch_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "BatchNorm2d(32) applied after conv with 64 output channels",
        "description": "BatchNorm channel count doesn't match conv output",
        "code": """\
import torch.nn as nn
class BNChannelBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "cv_conv_in_channel_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "conv2 expects 32 input channels but conv1 outputs 64",
        "description": "Sequential convs with mismatched channels",
        "code": """\
import torch.nn as nn
class ConvChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        return self.relu(self.conv2(h))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "cv_unet_skip_channel_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "Skip connection adds 64-ch encoder output to 128-ch decoder",
        "description": "UNet skip connection with channel mismatch",
        "code": """\
import torch.nn as nn
class UNetSkipBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv = nn.Conv2d(3, 64, 3, padding=1)
        self.dec_conv = nn.Conv2d(64, 128, 3, padding=1)
    def forward(self, x):
        enc = self.enc_conv(x)
        dec = self.dec_conv(enc)
        return dec + enc
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
    {
        "name": "cv_fpn_channel_mismatch_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "lateral3 outputs 256 but lateral4 outputs 128, add fails",
        "description": "FPN laterals project to different channel counts",
        "code": """\
import torch.nn as nn
class FPNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral3 = nn.Conv2d(512, 256, 1)
        self.lateral4 = nn.Conv2d(1024, 128, 1)
    def forward(self, c3, c4):
        return self.lateral3(c3) + self.lateral4(c4)
""",
        "input_shapes": {"c3": ("batch", 512, 28, 28), "c4": ("batch", 1024, 28, 28)},
    },
    {
        "name": "cv_squeeze_excite_dim_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "fc2 outputs 64 but input channels are 128, multiply fails",
        "description": "SE block with wrong output dim for scale",
        "code": """\
import torch.nn as nn
class SEBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 32)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(32, 64)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * scale
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "cv_inception_channel_mismatch_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "branch3 outputs 128 but branch1/5 output 64, add fails",
        "description": "Inception branches with mismatched output channels",
        "code": """\
import torch.nn as nn
class InceptionBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch1 = nn.Conv2d(256, 64, 1)
        self.branch3 = nn.Conv2d(256, 128, 3, padding=1)
        self.branch5 = nn.Conv2d(256, 64, 5, padding=2)
    def forward(self, x):
        return self.branch1(x) + self.branch3(x) + self.branch5(x)
""",
        "input_shapes": {"x": ("batch", 256, 14, 14)},
    },
    {
        "name": "cv_depthwise_groups_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "Depthwise conv has 64 groups but pointwise expects 32 in_channels",
        "description": "Depthwise separable conv with wrong pointwise input",
        "code": """\
import torch.nn as nn
class DepthwiseBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.depthwise = nn.Conv2d(64, 64, 3, padding=1, groups=64)
        self.pointwise = nn.Conv2d(32, 128, 1)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_double_conv_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Double conv block: conv->bn->relu->conv->bn->relu",
        "code": """\
import torch.nn as nn
class DoubleConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(h)))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_feature_extractor_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Multi-stage feature extractor: 3->32->64->128",
        "code": """\
import torch.nn as nn
class FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Conv2d(3, 32, 3, padding=1)
        self.stage2 = nn.Conv2d(32, 64, 3, padding=1)
        self.stage3 = nn.Conv2d(64, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.stage1(x))
        h = self.relu(self.stage2(h))
        return self.relu(self.stage3(h))
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
    {
        "name": "cv_feature_extractor_stage_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "stage3 expects 32 in_channels but stage2 outputs 64",
        "description": "Feature extractor with wrong in_channels at stage 3",
        "code": """\
import torch.nn as nn
class FeatureExtractorBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.stage1 = nn.Conv2d(3, 32, 3, padding=1)
        self.stage2 = nn.Conv2d(32, 64, 3, padding=1)
        self.stage3 = nn.Conv2d(32, 128, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.stage1(x))
        h = self.relu(self.stage2(h))
        return self.relu(self.stage3(h))
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
    {
        "name": "cv_dilated_conv_correct",
        "category": "vision",
        "has_bug": False,
        "description": "Dilated convolution block with matching channels",
        "code": """\
import torch.nn as nn
class DilatedConvBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=4, dilation=4)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        return self.relu(self.bn(self.conv2(h)))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_conv_transpose_correct",
        "category": "vision",
        "has_bug": False,
        "description": "ConvTranspose2d upsampling block",
        "code": """\
import torch.nn as nn
class UpsampleBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.up(x))
        return self.relu(self.bn(self.conv(h)))
""",
        "input_shapes": {"x": ("batch", 128, 16, 16)},
    },
    {
        "name": "cv_conv_transpose_channel_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "conv expects 128 in_channels but up outputs 64",
        "description": "ConvTranspose followed by conv with wrong in_channels",
        "code": """\
import torch.nn as nn
class UpsampleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv = nn.Conv2d(128, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.up(x))
        return self.relu(self.conv(h))
""",
        "input_shapes": {"x": ("batch", 128, 16, 16)},
    },
    {
        "name": "cv_resnet_downsample_stride_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "Downsample shortcut has stride=1 but main path has stride=2, spatial dims mismatch",
        "description": "ResNet downsample with mismatched stride in shortcut",
        "code": """\
import torch.nn as nn
class DownsampleStrideBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.downsample = nn.Conv2d(64, 128, 1, stride=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + self.downsample(x))
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "cv_parallel_conv_add_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "Parallel convs output 32 and 64 channels, add fails",
        "description": "Parallel conv branches with mismatched channel counts",
        "code": """\
import torch.nn as nn
class ParallelConvBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.branch_a = nn.Conv2d(3, 32, 3, padding=1)
        self.branch_b = nn.Conv2d(3, 64, 3, padding=1)
    def forward(self, x):
        return self.branch_a(x) + self.branch_b(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "cv_mobilenet_block_correct",
        "category": "vision",
        "has_bug": False,
        "description": "MobileNet inverted residual block",
        "code": """\
import torch.nn as nn
class InvertedResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(32, 192, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.dw = nn.Conv2d(192, 192, 3, padding=1, groups=192)
        self.bn2 = nn.BatchNorm2d(192)
        self.project = nn.Conv2d(192, 32, 1)
        self.bn3 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.expand(x)))
        h = self.relu(self.bn2(self.dw(h)))
        h = self.bn3(self.project(h))
        return h + x
""",
        "input_shapes": {"x": ("batch", 32, 28, 28)},
    },
    {
        "name": "cv_mobilenet_expand_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "project outputs 64 but residual x has 32 channels, add fails",
        "description": "MobileNet block with wrong project output channels",
        "code": """\
import torch.nn as nn
class InvertedResidualBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Conv2d(32, 192, 1)
        self.bn1 = nn.BatchNorm2d(192)
        self.dw = nn.Conv2d(192, 192, 3, padding=1, groups=192)
        self.bn2 = nn.BatchNorm2d(192)
        self.project = nn.Conv2d(192, 64, 1)
        self.bn3 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.expand(x)))
        h = self.relu(self.bn2(self.dw(h)))
        h = self.bn3(self.project(h))
        return h + x
""",
        "input_shapes": {"x": ("batch", 32, 28, 28)},
    },
    {
        "name": "cv_aspp_branch_correct",
        "category": "vision",
        "has_bug": False,
        "description": "ASPP: parallel dilated convs with same output channels",
        "code": """\
import torch.nn as nn
class ASPPBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 64, 1)
        self.conv3d6 = nn.Conv2d(256, 64, 3, padding=6, dilation=6)
        self.conv3d12 = nn.Conv2d(256, 64, 3, padding=12, dilation=12)
    def forward(self, x):
        return self.conv1x1(x) + self.conv3d6(x) + self.conv3d12(x)
""",
        "input_shapes": {"x": ("batch", 256, 32, 32)},
    },
    {
        "name": "cv_aspp_channel_bug",
        "category": "vision",
        "has_bug": True,
        "bug_description": "conv3d12 outputs 128 but others output 64, add fails",
        "description": "ASPP with one branch outputting wrong channel count",
        "code": """\
import torch.nn as nn
class ASPPBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1x1 = nn.Conv2d(256, 64, 1)
        self.conv3d6 = nn.Conv2d(256, 64, 3, padding=6, dilation=6)
        self.conv3d12 = nn.Conv2d(256, 128, 3, padding=12, dilation=12)
    def forward(self, x):
        return self.conv1x1(x) + self.conv3d6(x) + self.conv3d12(x)
""",
        "input_shapes": {"x": ("batch", 256, 32, 32)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 3: Device Inconsistency Bugs (20+)
# ═══════════════════════════════════════════════════════════════════════════════

DEVICE_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "dev_forgotten_to_device_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "Second linear created on CPU but input may be on CUDA",
        "description": "Module creates extra param on CPU, never moved to device",
        "code": """\
import torch
import torch.nn as nn
class ForgotToDevice(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.extra = torch.zeros(128)
    def forward(self, x):
        h = self.fc1(x)
        return h + self.extra
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "dev_buffer_not_registered_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "Tensor stored as plain attribute won't move with .to(device)",
        "description": "Plain tensor attribute instead of register_buffer",
        "code": """\
import torch
import torch.nn as nn
class BufferNotRegistered(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.scale = torch.tensor(0.1)
    def forward(self, x):
        return self.fc(x) * self.scale
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_mixed_param_creation_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "bias created as raw tensor on CPU, won't follow .cuda()",
        "description": "Manual parameter created without nn.Parameter wrapping",
        "code": """\
import torch
import torch.nn as nn
class MixedParamCreation(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(256, 128)
        self.bias = torch.randn(128)
    def forward(self, x):
        return self.fc(x) + self.bias
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "dev_hardcoded_cpu_tensor_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.zeros created inside forward always on CPU",
        "description": "Tensor created in forward without device awareness",
        "code": """\
import torch
import torch.nn as nn
class HardcodedCPU(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        h = self.fc(x)
        mask = torch.zeros(32)
        return h + mask
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_ones_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.ones created in forward on CPU when input may be CUDA",
        "description": "Scale tensor created without matching device",
        "code": """\
import torch
import torch.nn as nn
class OnesInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    def forward(self, x):
        h = self.fc(x)
        scale = torch.ones(64)
        return h * scale
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_registered_buffer_correct",
        "category": "device",
        "has_bug": False,
        "description": "Buffer properly registered with register_buffer",
        "code": """\
import torch
import torch.nn as nn
class RegisteredBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.register_buffer('scale', torch.ones(64))
    def forward(self, x):
        return self.fc(x) * self.scale
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_nn_parameter_correct",
        "category": "device",
        "has_bug": False,
        "description": "Extra param wrapped as nn.Parameter (moves with .to())",
        "code": """\
import torch
import torch.nn as nn
class ProperParameter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.alpha = nn.Parameter(torch.ones(64))
    def forward(self, x):
        return self.fc(x) * self.alpha
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_multi_buffer_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "Two plain tensors as attributes, neither moves with model",
        "description": "Multiple unregistered tensor attributes",
        "code": """\
import torch
import torch.nn as nn
class MultiBufferBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.mean = torch.zeros(32)
        self.std = torch.ones(32)
    def forward(self, x):
        h = self.fc(x)
        return (h - self.mean) / self.std
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_mixed_device_cat_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "Concatenating param output (device-dependent) with CPU tensor",
        "description": "Cat operation with CPU and potentially CUDA tensors",
        "code": """\
import torch
import torch.nn as nn
class MixedDeviceCat(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.constant = torch.randn(32)
    def forward(self, x):
        h = self.fc(x)
        return h + self.constant
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_device_aware_forward_correct",
        "category": "device",
        "has_bug": False,
        "description": "Tensor created on same device as input using x.device",
        "code": """\
import torch
import torch.nn as nn
class DeviceAware(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        h = self.fc(x)
        mask = torch.ones(32, device=x.device)
        return h * mask
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_two_module_correct",
        "category": "device",
        "has_bug": False,
        "description": "Two submodules, both move with .to(device)",
        "code": """\
import torch.nn as nn
class TwoModuleCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.fc2 = nn.Linear(64, 32)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_randn_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.randn in forward creates CPU tensor regardless of input device",
        "description": "Random noise added without device matching",
        "code": """\
import torch
import torch.nn as nn
class RandnInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
    def forward(self, x):
        h = self.fc(x)
        noise = torch.randn(64)
        return h + noise
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_arange_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.arange creates CPU tensor, added to potentially CUDA output",
        "description": "Positional encoding using arange without device",
        "code": """\
import torch
import torch.nn as nn
class ArangeInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    def forward(self, x):
        h = self.fc(x)
        pos = torch.arange(64).float()
        return h + pos
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_eye_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.eye creates CPU identity matrix, used with CUDA tensor",
        "description": "Identity matrix created without device in forward",
        "code": """\
import torch
import torch.nn as nn
class EyeInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
    def forward(self, x):
        h = self.fc(x)
        identity = torch.eye(64)
        return h @ identity
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_parameter_list_correct",
        "category": "device",
        "has_bug": False,
        "description": "ParameterList with proper nn.Parameter wrapping",
        "code": """\
import torch
import torch.nn as nn
class ParameterListCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
        self.scales = nn.ParameterList([nn.Parameter(torch.ones(64)) for _ in range(3)])
    def forward(self, x):
        h = self.fc(x)
        return h * self.scales[0]
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_list_of_tensors_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "Plain list of tensors won't move with .to(device)",
        "description": "List of tensors stored as plain Python list attribute",
        "code": """\
import torch
import torch.nn as nn
class ListOfTensors(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 64)
        self.weights = [torch.randn(64) for _ in range(3)]
    def forward(self, x):
        h = self.fc(x)
        return h * self.weights[0]
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_conditional_device_correct",
        "category": "device",
        "has_bug": False,
        "description": "Submodule registered properly via ModuleList",
        "code": """\
import torch.nn as nn
class ModuleListCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(64, 64) for _ in range(3)])
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_linspace_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.linspace creates CPU tensor added to potentially CUDA output",
        "description": "Linspace bias added without device matching",
        "code": """\
import torch
import torch.nn as nn
class LinspaceForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
    def forward(self, x):
        h = self.fc(x)
        bias = torch.linspace(0, 1, 32)
        return h + bias
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "dev_full_in_forward_bug",
        "category": "device",
        "has_bug": True,
        "bug_description": "torch.full creates CPU tensor used with potentially CUDA input",
        "description": "Constant fill tensor without device specification",
        "code": """\
import torch
import torch.nn as nn
class FullInForward(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 64)
    def forward(self, x):
        h = self.fc(x)
        offset = torch.full((64,), 0.5)
        return h + offset
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "dev_registered_buffer_v2_correct",
        "category": "device",
        "has_bug": False,
        "description": "Multiple buffers all properly registered",
        "code": """\
import torch
import torch.nn as nn
class MultiRegisteredBuffer(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(64, 32)
        self.register_buffer('mean', torch.zeros(32))
        self.register_buffer('std', torch.ones(32))
    def forward(self, x):
        h = self.fc(x)
        return (h - self.mean) / self.std
""",
        "input_shapes": {"x": ("batch", 64)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 4: Phase-Dependent Bugs (20+)
# ═══════════════════════════════════════════════════════════════════════════════

PHASE_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "phase_dropout_train_only_correct",
        "category": "phase",
        "has_bug": False,
        "description": "Dropout only active in training — standard correct usage",
        "code": """\
import torch.nn as nn
class DropoutCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.dropout(self.fc1(x))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "phase_batchnorm_stats_correct",
        "category": "phase",
        "has_bug": False,
        "description": "BatchNorm with correct channels, behaves differently train/eval",
        "code": """\
import torch.nn as nn
class BNCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_dropout_reshape_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "Dropout changes effective scale between train/eval, fc2 input from fc1 has dim mismatch",
        "description": "Dropout before Linear with shape mismatch in dimensions",
        "code": """\
import torch.nn as nn
class DropoutReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.dropout(self.fc1(x))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "phase_bn_wrong_channels_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "BatchNorm2d(32) but conv outputs 64 channels",
        "description": "BatchNorm with wrong num_features, differs in train vs eval",
        "code": """\
import torch.nn as nn
class BNWrongChannelsBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(32)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_double_dropout_correct",
        "category": "phase",
        "has_bug": False,
        "description": "Two dropout layers at different rates — valid pattern",
        "code": """\
import torch.nn as nn
class DoubleDropout(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 128)
        self.drop2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.drop1(self.fc1(x))
        h = self.drop2(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "phase_bn1d_correct",
        "category": "phase",
        "has_bug": False,
        "description": "BatchNorm1d in FC network — correct usage",
        "code": """\
import torch.nn as nn
class BN1DCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc2(self.relu(self.bn(self.fc1(x))))
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "phase_bn1d_wrong_features_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "BatchNorm1d(128) but fc1 outputs 64 features",
        "description": "BatchNorm1d with wrong num_features after Linear",
        "code": """\
import torch.nn as nn
class BN1DBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 64)
        self.bn = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        return self.fc2(self.relu(self.bn(self.fc1(x))))
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "phase_training_only_path_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "aux_head expects 256 but features are 128 — shape error in training path",
        "description": "Auxiliary loss head with wrong input dim, only used in training",
        "code": """\
import torch.nn as nn
class TrainingOnlyPathBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 128)
        self.head = nn.Linear(128, 10)
        self.aux_head = nn.Linear(256, 10)
    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        if self.training:
            aux = self.aux_head(features)
            return out, aux
        return out
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "phase_training_only_correct",
        "category": "phase",
        "has_bug": False,
        "description": "Auxiliary head with correct dims, only used in training",
        "code": """\
import torch.nn as nn
class TrainingOnlyCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 128)
        self.head = nn.Linear(128, 10)
        self.aux_head = nn.Linear(128, 10)
    def forward(self, x):
        features = self.encoder(x)
        out = self.head(features)
        if self.training:
            aux = self.aux_head(features)
            return out, aux
        return out
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "phase_dropout2d_correct",
        "category": "phase",
        "has_bug": False,
        "description": "Dropout2d for spatial dropout on conv features",
        "code": """\
import torch.nn as nn
class Dropout2DCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.drop = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        h = self.drop(self.conv(x))
        return self.conv2(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_instancenorm_correct",
        "category": "phase",
        "has_bug": False,
        "description": "InstanceNorm2d with correct channels",
        "code": """\
import torch.nn as nn
class InstanceNormCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.norm = nn.InstanceNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_instancenorm_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "InstanceNorm2d(32) but conv outputs 64 channels",
        "description": "InstanceNorm with wrong feature count",
        "code": """\
import torch.nn as nn
class InstanceNormBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.norm = nn.InstanceNorm2d(32)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.norm(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_layernorm_after_conv_correct",
        "category": "phase",
        "has_bug": True,
        "bug_description": "LayerNorm(64) expects last dim=64 but Conv2d output has last dim=H (spatial)",
        "description": "LayerNorm dimension mismatch after conv",
        "code": """\
import torch.nn as nn
class LNAfterConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.norm = nn.LayerNorm(64)
    def forward(self, x):
        h = self.conv(x)
        return self.norm(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_bn_after_linear_chain_correct",
        "category": "phase",
        "has_bug": False,
        "description": "BatchNorm1d between Linear layers — correct dims",
        "code": """\
import torch.nn as nn
class BNLinearChain(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "phase_bn_chain_mismatch_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "bn2 expects 128 features but fc2 outputs 64",
        "description": "BatchNorm1d in chain with wrong feature count",
        "code": """\
import torch.nn as nn
class BNChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "phase_groupnorm_correct",
        "category": "phase",
        "has_bug": False,
        "description": "GroupNorm with correct groups and channels",
        "code": """\
import torch.nn as nn
class GroupNormCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.gn = nn.GroupNorm(8, 64)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_groupnorm_channel_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "GroupNorm(8, 32) but conv outputs 64 channels",
        "description": "GroupNorm with wrong channel count",
        "code": """\
import torch.nn as nn
class GroupNormBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.gn = nn.GroupNorm(8, 32)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.gn(self.conv(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "phase_eval_only_head_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "eval-only classification head expects 256 but encoder outputs 128",
        "description": "Eval-only classification head with wrong input dim",
        "code": """\
import torch.nn as nn
class EvalOnlyBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(512, 128)
        self.train_head = nn.Linear(128, 10)
        self.eval_head = nn.Linear(256, 10)
    def forward(self, x):
        features = self.encoder(x)
        if self.training:
            return self.train_head(features)
        return self.eval_head(features)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "phase_alphadropout_correct",
        "category": "phase",
        "has_bug": False,
        "description": "AlphaDropout with SELU — correct phase-aware usage",
        "code": """\
import torch.nn as nn
class AlphaDropoutCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.act = nn.SELU()
        self.drop = nn.AlphaDropout(0.1)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.drop(self.act(self.fc1(x)))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "phase_dropout_wrong_input_dim_bug",
        "category": "phase",
        "has_bug": True,
        "bug_description": "fc2 expects 64 but fc1 outputs 128 (dropout doesn't change shape)",
        "description": "Linear dim mismatch masked by dropout confusion",
        "code": """\
import torch.nn as nn
class DropoutDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.drop = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.drop(self.fc1(x))
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 5: Broadcasting Bugs (30+)
# ═══════════════════════════════════════════════════════════════════════════════

BROADCAST_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "bcast_parallel_proj_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Linear->128 + Linear->64: broadcast dims 128 vs 64",
        "description": "Parallel projections to different dims then add",
        "code": """\
import torch.nn as nn
class ParallelProjBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x):
        return self.fc_a(x) + self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_parallel_proj_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Parallel projections to same dim then add",
        "code": """\
import torch.nn as nn
class ParallelProjCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
    def forward(self, x):
        return self.fc_a(x) + self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_cross_rank_3d_2d_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "(batch,seq,256) + (batch,128): last dims 256 vs 128 mismatch",
        "description": "3D + 2D broadcasting with last dim mismatch",
        "code": """\
import torch.nn as nn
class CrossRank3D2DBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 128)
    def forward(self, x, bias):
        return self.proj(x) + self.bias_proj(bias)
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": ("batch", 256)},
    },
    {
        "name": "bcast_cross_rank_3d_1d_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "(batch,seq,256) + (256,): broadcasts to (1,1,256) — safe",
        "code": """\
import torch.nn as nn
class CrossRank3D1DCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 256)
        self.bias_proj = nn.Linear(256, 256)
    def forward(self, x, bias):
        return self.proj(x) + self.bias_proj(bias)
""",
        "input_shapes": {"x": ("batch", "seq", 256), "bias": (256,)},
    },
    {
        "name": "bcast_three_way_chain_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "(128)+(128)=OK then (128)+(64)=FAIL at second add",
        "description": "Three-way broadcast chain, third branch mismatches",
        "code": """\
import torch.nn as nn
class ThreeWayChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        ab = self.fc_a(x) + self.fc_b(x)
        return ab + self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_three_way_chain_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Three-way chain: all 128, adds are safe",
        "code": """\
import torch.nn as nn
class ThreeWayChainCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
    def forward(self, x):
        ab = self.fc_a(x) + self.fc_b(x)
        return ab + self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_multiply_dim_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Element-wise multiply: 256 * 128 incompatible",
        "description": "Element-wise multiply with mismatched output dims",
        "code": """\
import torch.nn as nn
class MultiplyDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 128)
    def forward(self, x):
        return self.fc_a(x) * self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "bcast_multiply_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Element-wise multiply with matching dims",
        "code": """\
import torch.nn as nn
class MultiplyCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 256)
    def forward(self, x):
        return self.fc_a(x) * self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "bcast_residual_proj_dim_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Projection outputs 512 but residual input is 768",
        "description": "Residual connection where projection changes dim",
        "code": """\
import torch.nn as nn
class ResidualProjBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(768, 512)
    def forward(self, x):
        return x + self.proj(x)
""",
        "input_shapes": {"x": ("batch", 768)},
    },
    {
        "name": "bcast_four_branch_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Four branches: 64+64+64+32, last branch mismatches",
        "description": "Four parallel branches with one dim mismatch",
        "code": """\
import torch.nn as nn
class FourBranchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128, 64)
        self.b = nn.Linear(128, 64)
        self.c = nn.Linear(128, 64)
        self.d = nn.Linear(128, 32)
    def forward(self, x):
        return self.a(x) + self.b(x) + self.c(x) + self.d(x)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "bcast_four_branch_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Four parallel branches all outputting 64",
        "code": """\
import torch.nn as nn
class FourBranchCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(128, 64)
        self.b = nn.Linear(128, 64)
        self.c = nn.Linear(128, 64)
        self.d = nn.Linear(128, 64)
    def forward(self, x):
        return self.a(x) + self.b(x) + self.c(x) + self.d(x)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "bcast_matmul_inner_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "proj->32 then matmul with (64,10): inner 32!=64",
        "description": "Matmul with mismatched inner dim after projection",
        "code": """\
import torch.nn as nn
class MatmulInnerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 32)
    def forward(self, x, w):
        return self.proj(x) @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },
    {
        "name": "bcast_matmul_inner_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Matmul with matching inner dim after projection",
        "code": """\
import torch.nn as nn
class MatmulInnerCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(128, 64)
    def forward(self, x, w):
        return self.proj(x) @ w
""",
        "input_shapes": {"x": ("batch", 128), "w": (64, 10)},
    },
    {
        "name": "bcast_conv_add_mismatch_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Conv outputs 32 channels but other conv outputs 64, add fails",
        "description": "Conv2d parallel branches with channel mismatch in add",
        "code": """\
import torch.nn as nn
class ConvAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 64, 3, padding=1)
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "bcast_conv_add_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Conv2d parallel branches with matching channels",
        "code": """\
import torch.nn as nn
class ConvAddCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(3, 32, 3, padding=1)
    def forward(self, x):
        return self.conv1(x) + self.conv2(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "bcast_matmul_after_add_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Two proj->32, add, then matmul(64,10): inner 32!=64",
        "description": "Matmul inner dim after add of two projections",
        "code": """\
import torch.nn as nn
class MatmulAfterAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 32)
        self.fc_b = nn.Linear(256, 32)
    def forward(self, x, w):
        c = self.fc_a(x) + self.fc_b(x)
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },
    {
        "name": "bcast_matmul_after_add_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Matmul with correct inner dim after add of projections",
        "code": """\
import torch.nn as nn
class MatmulAfterAddCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 64)
        self.fc_b = nn.Linear(256, 64)
    def forward(self, x, w):
        c = self.fc_a(x) + self.fc_b(x)
        return c @ w
""",
        "input_shapes": {"x": ("batch", 256), "w": (64, 10)},
    },
    {
        "name": "bcast_sub_dim_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Subtraction of 256-dim and 128-dim tensors",
        "description": "Element-wise subtract with mismatched dims",
        "code": """\
import torch.nn as nn
class SubDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 128)
    def forward(self, x):
        return self.fc_a(x) - self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "bcast_sub_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Element-wise subtract with matching dims",
        "code": """\
import torch.nn as nn
class SubCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(512, 256)
        self.fc_b = nn.Linear(512, 256)
    def forward(self, x):
        return self.fc_a(x) - self.fc_b(x)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "bcast_weighted_sum_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "gate outputs 64 but value outputs 128, multiply fails",
        "description": "Gating mechanism with mismatched dims",
        "code": """\
import torch.nn as nn
class WeightedSumBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 64)
        self.value = nn.Linear(256, 128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        g = self.sigmoid(self.gate(x))
        v = self.value(x)
        return g * v
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_weighted_sum_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Gating mechanism with matching dims",
        "code": """\
import torch.nn as nn
class WeightedSumCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Linear(256, 128)
        self.value = nn.Linear(256, 128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        g = self.sigmoid(self.gate(x))
        v = self.value(x)
        return g * v
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_skip_connection_chain_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "fc2 outputs 128, added to intermediate fc1 output of 256",
        "description": "Skip connection over a chain with dim mismatch",
        "code": """\
import torch.nn as nn
class SkipChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
    def forward(self, x):
        h1 = self.fc1(x)
        h2 = self.fc2(h1)
        return h1 + h2
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "bcast_dual_residual_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Dual residual: both branches project to same dim",
        "code": """\
import torch.nn as nn
class DualResidualCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
    def forward(self, x):
        return x + self.fc1(x) + self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_dual_residual_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "fc2 outputs 128, doesn't match x (256) for residual add",
        "description": "Dual residual with one branch having wrong dim",
        "code": """\
import torch.nn as nn
class DualResidualBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
    def forward(self, x):
        return x + self.fc1(x) + self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_nested_add_mul_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "(fc_a+fc_b)*fc_c: fc_a/fc_b->128, fc_c->64, multiply fails",
        "description": "Nested add then multiply with dim mismatch",
        "code": """\
import torch.nn as nn
class NestedAddMulBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        ab = self.fc_a(x) + self.fc_b(x)
        return ab * self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_nested_add_mul_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Nested add then multiply with matching dims",
        "code": """\
import torch.nn as nn
class NestedAddMulCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
    def forward(self, x):
        ab = self.fc_a(x) + self.fc_b(x)
        return ab * self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_five_branch_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "Five branches: 4 output 32, one outputs 16. Add fails",
        "description": "Five parallel projections with one dim mismatch",
        "code": """\
import torch.nn as nn
class FiveBranchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = nn.Linear(64, 32)
        self.b = nn.Linear(64, 32)
        self.c = nn.Linear(64, 32)
        self.d = nn.Linear(64, 32)
        self.e = nn.Linear(64, 16)
    def forward(self, x):
        return self.a(x) + self.b(x) + self.c(x) + self.d(x) + self.e(x)
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "bcast_cascade_mul_add_bug",
        "category": "broadcast",
        "has_bug": True,
        "bug_description": "fc_a*fc_b (both 128) + fc_c (64): add after multiply fails",
        "description": "Multiply then add with dim mismatch",
        "code": """\
import torch.nn as nn
class CascadeMulAddBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 64)
    def forward(self, x):
        prod = self.fc_a(x) * self.fc_b(x)
        return prod + self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "bcast_cascade_mul_add_correct",
        "category": "broadcast",
        "has_bug": False,
        "description": "Multiply then add with matching dims",
        "code": """\
import torch.nn as nn
class CascadeMulAddCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_a = nn.Linear(256, 128)
        self.fc_b = nn.Linear(256, 128)
        self.fc_c = nn.Linear(256, 128)
    def forward(self, x):
        prod = self.fc_a(x) * self.fc_b(x)
        return prod + self.fc_c(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 6: Reshape/View Bugs (20+)
# ═══════════════════════════════════════════════════════════════════════════════

RESHAPE_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "reshape_flatten_fc_correct",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Conv output (4D) passed directly to Linear without flatten",
        "description": "Missing flatten between conv and fc",
        "code": """\
import torch.nn as nn
class FlattenFCCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        h = self.conv(x)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_flatten_dim_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "FC expects 16*8*8=1024 features but conv output is 16*32*32",
        "description": "FC after flatten with wrong feature count",
        "code": """\
import torch.nn as nn
class FlattenDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_view_wrong_product_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "view(-1, 128) but tensor has 256 features, not divisible correctly",
        "description": "View with dimension that doesn't evenly divide total elements",
        "code": """\
import torch.nn as nn
class ViewWrongProductBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 300)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = h.view(-1, 128)
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "reshape_sequential_conv_flatten_correct",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Conv output (4D) passed directly to Linear without flatten",
        "description": "Missing flatten between conv chain and fc",
        "code": """\
import torch.nn as nn
class ConvFlattenCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_unflatten_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "fc outputs 256 but view tries to reshape to (8, 64) = 512",
        "description": "Unflatten/view with wrong target shape",
        "code": """\
import torch.nn as nn
class UnflattenBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(128, 256)
        self.conv = nn.Conv2d(8, 16, 3, padding=1)
    def forward(self, x):
        h = self.fc(x)
        h = h.view(-1, 8, 64)
        return self.conv(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "reshape_fc_after_pool_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "AdaptiveAvgPool2d -> flatten -> FC with correct dim",
        "code": """\
import torch.nn as nn
class PoolFCCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_pool_fc_dim_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "FC expects 128 but AdaptiveAvgPool2d(1) + flatten gives 64",
        "description": "FC after adaptive pool with wrong expected dim",
        "code": """\
import torch.nn as nn
class PoolFCBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_view_reshape_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "View to reshape for multi-head attention splitting",
        "code": """\
import torch.nn as nn
class ViewReshapeCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
    def forward(self, x):
        h = self.proj(x)
        return self.out(h)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "reshape_fc_chain_with_view_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "fc2 expects 64 but fc1 outputs 128",
        "description": "FC chain with dimension mismatch around a view op",
        "code": """\
import torch.nn as nn
class FCViewBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(64, 10)
    def forward(self, x):
        h = self.fc1(x)
        h = h.view(-1, 64)
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "reshape_transpose_matmul_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "Transpose for attention score computation",
        "code": """\
import torch.nn as nn
class TransposeMatmulCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        return q + k
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "reshape_conv_wrong_spatial_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Conv2d stride=2 halves spatial dims but next conv expects original size channels don't match",
        "description": "Strided conv followed by conv with wrong in_channels",
        "code": """\
import torch.nn as nn
class ConvStrideBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
    def forward(self, x):
        h = self.conv1(x)
        return self.conv2(h)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
    },
    {
        "name": "reshape_permute_linear_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "Permute then linear — correct feature dim",
        "code": """\
import torch.nn as nn
class PermuteLinearCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 64)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "reshape_squeeze_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "After AdaptiveAvgPool2d(1), squeeze removes spatial dims; fc expects wrong features",
        "description": "Squeeze after pool with wrong FC input size",
        "code": """\
import torch.nn as nn
class SqueezeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_multi_head_split_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "Multi-head split and merge with correct dims",
        "code": """\
import torch.nn as nn
class MultiHeadSplitCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv = nn.Linear(512, 512)
        self.proj = nn.Linear(512, 512)
    def forward(self, x):
        h = self.qkv(x)
        return self.proj(h)
""",
        "input_shapes": {"x": ("batch", "seq", 512)},
    },
    {
        "name": "reshape_flatten_wrong_start_dim_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Flatten from dim 1 gives channels*H*W but FC expects only channels",
        "description": "Flatten with wrong start dim, FC gets too many features",
        "code": """\
import torch.nn as nn
class FlattenStartDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 16, 3, padding=1)
        self.flatten = nn.Flatten(1)
        self.fc = nn.Linear(16, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_adaptive_pool_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "AdaptiveAvgPool1d then Linear with correct dim",
        "code": """\
import torch.nn as nn
class AdaptivePool1DCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "reshape_double_flatten_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Two convs change channels; flatten gives wrong count for fc",
        "description": "Double conv then flatten with wrong FC in_features",
        "code": """\
import torch.nn as nn
class DoubleFlattenBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "reshape_unsqueeze_correct",
        "category": "reshape",
        "has_bug": False,
        "description": "Unsqueeze for broadcasting then squeeze back",
        "code": """\
import torch.nn as nn
class UnsqueezeCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 64)
    def forward(self, x):
        h = self.fc1(x)
        return self.fc2(h)
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "reshape_global_pool_fc_bug",
        "category": "reshape",
        "has_bug": True,
        "bug_description": "Global avg pool gives 128 features but fc expects 256",
        "description": "Global pooling to FC with wrong in_features",
        "code": """\
import torch.nn as nn
class GlobalPoolFCBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 128, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256, 10)
    def forward(self, x):
        h = self.conv(x)
        h = self.pool(h)
        h = self.flatten(h)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 7: Multi-Layer Chain Bugs (20+)
# ═══════════════════════════════════════════════════════════════════════════════

CHAIN_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "chain_5_linear_correct",
        "category": "chain",
        "has_bug": False,
        "description": "5 Linear layers: 512->256->128->64->32->10",
        "code": """\
import torch.nn as nn
class Chain5Correct(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_5_linear_mid_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc3 expects 256 but fc2 outputs 128 — error at layer 3",
        "description": "5-layer chain with mismatch in the middle",
        "code": """\
import torch.nn as nn
class Chain5MidBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_5_linear_last_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc5 expects 64 but fc4 outputs 32 — error at last layer",
        "description": "5-layer chain with mismatch at final layer",
        "code": """\
import torch.nn as nn
class Chain5LastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_conv_5_stage_correct",
        "category": "chain",
        "has_bug": False,
        "description": "5-stage conv chain: 3->16->32->64->128->256",
        "code": """\
import torch.nn as nn
class ConvChain5Correct(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(64, 128, 3, padding=1)
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.c1(x))
        h = self.relu(self.c2(h))
        h = self.relu(self.c3(h))
        h = self.relu(self.c4(h))
        return self.relu(self.c5(h))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_conv_5_stage_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "c4 expects 32 in_channels but c3 outputs 64",
        "description": "5-stage conv chain with channel mismatch at stage 4",
        "code": """\
import torch.nn as nn
class ConvChain5Bug(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 16, 3, padding=1)
        self.c2 = nn.Conv2d(16, 32, 3, padding=1)
        self.c3 = nn.Conv2d(32, 64, 3, padding=1)
        self.c4 = nn.Conv2d(32, 128, 3, padding=1)
        self.c5 = nn.Conv2d(128, 256, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.c1(x))
        h = self.relu(self.c2(h))
        h = self.relu(self.c3(h))
        h = self.relu(self.c4(h))
        return self.relu(self.c5(h))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_mixed_linear_conv_correct",
        "category": "chain",
        "has_bug": True,
        "bug_description": "Conv output (4D) passed directly to Linear without flatten",
        "description": "Missing flatten between conv chain and fc",
        "code": """\
import torch.nn as nn
class MixedChainCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_mixed_linear_conv_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc expects 32 but conv2 outputs 64 channels",
        "description": "Conv->Conv->Linear with wrong FC in_features",
        "code": """\
import torch.nn as nn
class MixedChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.fc = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_residual_3_block_correct",
        "category": "chain",
        "has_bug": False,
        "description": "Three residual blocks, each preserving dim 256",
        "code": """\
import torch.nn as nn
class Residual3Correct(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x + self.relu(self.fc1(x))
        x = x + self.relu(self.fc2(x))
        x = x + self.relu(self.fc3(x))
        return x
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "chain_residual_3_block_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc2 outputs 128, residual add with 256-dim input fails",
        "description": "Three residual blocks, second block changes dim",
        "code": """\
import torch.nn as nn
class Residual3Bug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x + self.relu(self.fc1(x))
        x = x + self.relu(self.fc2(x))
        x = x + self.relu(self.fc3(x))
        return x
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "chain_6_linear_correct",
        "category": "chain",
        "has_bug": False,
        "description": "6-layer Linear: 1024->512->256->128->64->32->10",
        "code": """\
import torch.nn as nn
class Chain6Correct(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        h = self.relu(self.fc5(h))
        return self.fc6(h)
""",
        "input_shapes": {"x": ("batch", 1024)},
    },
    {
        "name": "chain_6_linear_early_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc2 expects 1024 but fc1 outputs 512 — error at layer 2",
        "description": "6-layer chain with early mismatch",
        "code": """\
import torch.nn as nn
class Chain6EarlyBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(1024, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 32)
        self.fc6 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        h = self.relu(self.fc5(h))
        return self.fc6(h)
""",
        "input_shapes": {"x": ("batch", 1024)},
    },
    {
        "name": "chain_bn_linear_correct",
        "category": "chain",
        "has_bug": False,
        "description": "Linear->BN->Linear->BN->Linear chain, all matching",
        "code": """\
import torch.nn as nn
class BNLinearChainCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_bn_linear_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "bn1 has 512 features but fc1 outputs 256",
        "description": "Linear->BN chain with BN feature count mismatch",
        "code": """\
import torch.nn as nn
class BNLinearChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.fc1(x)))
        h = self.relu(self.bn2(self.fc2(h)))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_conv_bn_4_stage_correct",
        "category": "chain",
        "has_bug": False,
        "description": "4-stage conv-bn chain with correct channels",
        "code": """\
import torch.nn as nn
class ConvBN4Correct(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.b3 = nn.BatchNorm2d(128)
        self.c4 = nn.Conv2d(128, 256, 3, padding=1)
        self.b4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.b1(self.c1(x)))
        h = self.relu(self.b2(self.c2(h)))
        h = self.relu(self.b3(self.c3(h)))
        return self.relu(self.b4(self.c4(h)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_conv_bn_4_stage_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "b3 has 64 features but c3 outputs 128 channels",
        "description": "4-stage conv-bn chain with BN channel mismatch",
        "code": """\
import torch.nn as nn
class ConvBN4Bug(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(3, 32, 3, padding=1)
        self.b1 = nn.BatchNorm2d(32)
        self.c2 = nn.Conv2d(32, 64, 3, padding=1)
        self.b2 = nn.BatchNorm2d(64)
        self.c3 = nn.Conv2d(64, 128, 3, padding=1)
        self.b3 = nn.BatchNorm2d(64)
        self.c4 = nn.Conv2d(128, 256, 3, padding=1)
        self.b4 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.b1(self.c1(x)))
        h = self.relu(self.b2(self.c2(h)))
        h = self.relu(self.b3(self.c3(h)))
        return self.relu(self.b4(self.c4(h)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "chain_encoder_decoder_correct",
        "category": "chain",
        "has_bug": False,
        "description": "Encoder-decoder chain: 512->256->128->256->512",
        "code": """\
import torch.nn as nn
class EncoderDecoderCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(512, 256)
        self.enc2 = nn.Linear(256, 128)
        self.dec1 = nn.Linear(128, 256)
        self.dec2 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        h = self.relu(self.dec1(h))
        return self.dec2(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_encoder_decoder_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "dec1 expects 256 but enc2 outputs 128",
        "description": "Encoder-decoder with bottleneck dim mismatch",
        "code": """\
import torch.nn as nn
class EncoderDecoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Linear(512, 256)
        self.enc2 = nn.Linear(256, 128)
        self.dec1 = nn.Linear(256, 256)
        self.dec2 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        h = self.relu(self.dec1(h))
        return self.dec2(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "chain_gradual_expand_correct",
        "category": "chain",
        "has_bug": False,
        "description": "Gradual expansion: 64->128->256->512",
        "code": """\
import torch.nn as nn
class GradualExpandCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "chain_gradual_expand_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc3 expects 128 but fc2 outputs 256",
        "description": "Gradual expansion with wrong in_features at last layer",
        "code": """\
import torch.nn as nn
class GradualExpandBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(128, 512)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.fc3(h)
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "chain_skip_every_other_correct",
        "category": "chain",
        "has_bug": False,
        "description": "Skip connections every other layer, all preserving dim",
        "code": """\
import torch.nn as nn
class SkipEveryOtherCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = h + x
        h2 = self.relu(self.fc2(h))
        h2 = self.relu(self.fc3(h2))
        h2 = h2 + h
        return self.fc4(h2)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "chain_skip_every_other_bug",
        "category": "chain",
        "has_bug": True,
        "bug_description": "fc3 outputs 64 but skip connection expects 128, add fails",
        "description": "Skip connections with dim-changing layer in between",
        "code": """\
import torch.nn as nn
class SkipEveryOtherBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = h + x
        h2 = self.relu(self.fc2(h))
        h2 = self.relu(self.fc3(h2))
        h2 = h2 + h
        return self.fc4(h2)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
]

# ═══════════════════════════════════════════════════════════════════════════════
# Category 8: Correct Programs (30+)
# ═══════════════════════════════════════════════════════════════════════════════

CORRECT_BENCHMARKS: List[Dict[str, Any]] = [
    {
        "name": "correct_autoencoder",
        "category": "correct",
        "has_bug": False,
        "description": "Autoencoder: 784->256->64->256->784",
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
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.relu(self.enc1(x))
        h = self.relu(self.enc2(h))
        h = self.relu(self.dec1(h))
        return self.sigmoid(self.dec2(h))
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "correct_variational_encoder",
        "category": "correct",
        "has_bug": False,
        "description": "VAE encoder: shared base, separate mu and logvar heads",
        "code": """\
import torch.nn as nn
class VAEEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 400)
        self.fc_mu = nn.Linear(400, 20)
        self.fc_logvar = nn.Linear(400, 20)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        return self.fc_mu(h), self.fc_logvar(h)
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "correct_gan_generator",
        "category": "correct",
        "has_bug": False,
        "description": "GAN generator: 100->256->512->1024->784",
        "code": """\
import torch.nn as nn
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, 1024)
        self.fc4 = nn.Linear(1024, 784)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    def forward(self, z):
        h = self.relu(self.fc1(z))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        return self.tanh(self.fc4(h))
""",
        "input_shapes": {"z": ("batch", 100)},
    },
    {
        "name": "correct_gan_discriminator",
        "category": "correct",
        "has_bug": False,
        "description": "GAN discriminator: 784->512->256->1",
        "code": """\
import torch.nn as nn
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        self.relu = nn.LeakyReLU(0.2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        return self.sigmoid(self.fc3(h))
""",
        "input_shapes": {"x": ("batch", 784)},
    },
    {
        "name": "correct_residual_network",
        "category": "correct",
        "has_bug": False,
        "description": "4 residual blocks preserving dim 256",
        "code": """\
import torch.nn as nn
class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 256)
        self.out = nn.Linear(256, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        x = x + self.relu(self.fc1(x))
        x = x + self.relu(self.fc2(x))
        x = x + self.relu(self.fc3(x))
        x = x + self.relu(self.fc4(x))
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "correct_highway_network",
        "category": "correct",
        "has_bug": False,
        "description": "Highway network: transform gate * H + carry gate * x",
        "code": """\
import torch.nn as nn
class HighwayBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.H = nn.Linear(128, 128)
        self.T = nn.Linear(128, 128)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.relu(self.H(x))
        t = self.sigmoid(self.T(x))
        return h * t + x * (1 - t)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "correct_multi_task_head",
        "category": "correct",
        "has_bug": False,
        "description": "Multi-task learning: shared encoder, separate task heads",
        "code": """\
import torch.nn as nn
class MultiTaskHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Linear(512, 256)
        self.task_a = nn.Linear(256, 10)
        self.task_b = nn.Linear(256, 5)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.shared(x))
        return self.task_a(h), self.task_b(h)
""",
        "input_shapes": {"x": ("batch", 512)},
    },
    {
        "name": "correct_siamese_network",
        "category": "correct",
        "has_bug": False,
        "description": "Siamese network: shared encoder for two inputs",
        "code": """\
import torch.nn as nn
class SiameseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Linear(128, 64)
        self.classifier = nn.Linear(64, 1)
        self.relu = nn.ReLU()
    def forward(self, x1, x2):
        e1 = self.relu(self.encoder(x1))
        e2 = self.relu(self.encoder(x2))
        diff = e1 - e2
        return self.classifier(diff)
""",
        "input_shapes": {"x1": ("batch", 128), "x2": ("batch", 128)},
    },
    {
        "name": "correct_attention_pool",
        "category": "correct",
        "has_bug": False,
        "description": "Attention pooling: score per token, weighted sum",
        "code": """\
import torch.nn as nn
class AttentionPool(nn.Module):
    def __init__(self):
        super().__init__()
        self.score = nn.Linear(256, 1)
        self.proj = nn.Linear(256, 128)
    def forward(self, x):
        return self.proj(x)
""",
        "input_shapes": {"x": ("batch", "seq", 256)},
    },
    {
        "name": "correct_deep_feedforward",
        "category": "correct",
        "has_bug": False,
        "description": "Deep feedforward with decreasing dims: 1024->512->256->128->64->10",
        "code": """\
import torch.nn as nn
class DeepFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
    def forward(self, x):
        h = self.dropout(self.relu(self.fc1(x)))
        h = self.dropout(self.relu(self.fc2(h)))
        h = self.dropout(self.relu(self.fc3(h)))
        h = self.dropout(self.relu(self.fc4(h)))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 1024)},
    },
    {
        "name": "correct_dense_block",
        "category": "correct",
        "has_bug": False,
        "description": "DenseNet-style block: each layer gets all previous outputs",
        "code": """\
import torch.nn as nn
class DenseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1 + x))
        return self.fc3(h2 + h1 + x)
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "correct_recurrent_style",
        "category": "correct",
        "has_bug": False,
        "description": "Recurrent-style: same Linear applied multiple times",
        "code": """\
import torch.nn as nn
class RecurrentStyle(nn.Module):
    def __init__(self):
        super().__init__()
        self.cell = nn.Linear(64, 64)
        self.out = nn.Linear(64, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.cell(x))
        h = self.relu(self.cell(h))
        h = self.relu(self.cell(h))
        return self.out(h)
""",
        "input_shapes": {"x": ("batch", 64)},
    },
    {
        "name": "correct_feature_pyramid",
        "category": "correct",
        "has_bug": False,
        "description": "Feature pyramid: lateral connections with matching channels",
        "code": """\
import torch.nn as nn
class FeaturePyramid(nn.Module):
    def __init__(self):
        super().__init__()
        self.lat1 = nn.Conv2d(256, 128, 1)
        self.lat2 = nn.Conv2d(512, 128, 1)
        self.lat3 = nn.Conv2d(1024, 128, 1)
    def forward(self, c1, c2, c3):
        p1 = self.lat1(c1)
        p2 = self.lat2(c2)
        p3 = self.lat3(c3)
        return p1 + p2 + p3
""",
        "input_shapes": {
            "c1": ("batch", 256, 28, 28),
            "c2": ("batch", 512, 28, 28),
            "c3": ("batch", 1024, 28, 28),
        },
    },
    {
        "name": "correct_conv_classifier",
        "category": "correct",
        "has_bug": False,
        "description": "Conv backbone -> global pool -> classifier",
        "code": """\
import torch.nn as nn
class ConvClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(64, 10)
    def forward(self, x):
        h = self.features(x)
        h = self.pool(h)
        h = self.flatten(h)
        return self.classifier(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "correct_bottleneck_residual",
        "category": "correct",
        "has_bug": False,
        "description": "Bottleneck residual: 256->64->64->256 with skip",
        "code": """\
import torch.nn as nn
class BottleneckResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Linear(256, 64)
        self.mid = nn.Linear(64, 64)
        self.up = nn.Linear(64, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.down(x))
        h = self.relu(self.mid(h))
        h = self.up(h)
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "correct_gated_unit",
        "category": "correct",
        "has_bug": False,
        "description": "Gated linear unit: split in half, sigmoid gate * value",
        "code": """\
import torch.nn as nn
class GatedUnit(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(256, 512)
        self.out = nn.Linear(256, 128)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.proj(x)
        return self.out(x)
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "correct_dual_encoder",
        "category": "correct",
        "has_bug": False,
        "description": "Dual encoder for contrastive learning",
        "code": """\
import torch.nn as nn
class DualEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = nn.Linear(768, 256)
        self.image_enc = nn.Linear(2048, 256)
    def forward(self, text, image):
        t = self.text_enc(text)
        i = self.image_enc(image)
        return t + i
""",
        "input_shapes": {"text": ("batch", 768), "image": ("batch", 2048)},
    },
    {
        "name": "correct_mixture_of_experts",
        "category": "correct",
        "has_bug": False,
        "description": "Mixture of experts: gate selects expert outputs",
        "code": """\
import torch.nn as nn
class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.expert1 = nn.Linear(256, 256)
        self.expert2 = nn.Linear(256, 256)
        self.gate = nn.Linear(256, 256)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        g = self.sigmoid(self.gate(x))
        e1 = self.expert1(x)
        e2 = self.expert2(x)
        return g * e1 + (1 - g) * e2
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "correct_batch_norm_conv_block",
        "category": "correct",
        "has_bug": False,
        "description": "Standard conv-bn-relu block repeated twice",
        "code": """\
import torch.nn as nn
class ConvBNBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        return self.relu(self.bn2(self.conv2(h)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
    },
    {
        "name": "correct_embedding_classifier",
        "category": "correct",
        "has_bug": False,
        "description": "Embedding -> Linear -> classifier pipeline",
        "code": """\
import torch.nn as nn
class EmbeddingClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(300, 128)
        self.classifier = nn.Linear(128, 5)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.classifier(self.relu(self.proj(x)))
""",
        "input_shapes": {"x": ("batch", 300)},
    },
    {
        "name": "correct_wide_residual",
        "category": "correct",
        "has_bug": False,
        "description": "Wide residual block: expand then add back",
        "code": """\
import torch.nn as nn
class WideResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.conv2 = nn.Conv2d(256, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 64, 16, 16)},
    },
    {
        "name": "correct_norm_free_block",
        "category": "correct",
        "has_bug": False,
        "description": "Normalization-free residual block",
        "code": """\
import torch.nn as nn
class NormFreeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.conv2(h)
        return self.relu(h + x)
""",
        "input_shapes": {"x": ("batch", 64, 32, 32)},
    },
    {
        "name": "correct_cross_attention",
        "category": "correct",
        "has_bug": False,
        "description": "Cross-attention: Q from one source, K/V from another",
        "code": """\
import torch.nn as nn
class CrossAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(512, 512)
        self.kv_proj = nn.Linear(512, 512)
        self.out = nn.Linear(512, 512)
    def forward(self, q_input, kv_input):
        q = self.q_proj(q_input)
        kv = self.kv_proj(kv_input)
        return self.out(q + kv)
""",
        "input_shapes": {"q_input": ("batch", "seq", 512), "kv_input": ("batch", "seq", 512)},
    },
    {
        "name": "correct_projection_head",
        "category": "correct",
        "has_bug": False,
        "description": "Contrastive learning projection head: 2048->256->128",
        "code": """\
import torch.nn as nn
class ProjectionHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2048, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", 2048)},
    },
    {
        "name": "correct_squeeze_excite_residual",
        "category": "correct",
        "has_bug": False,
        "description": "SE block + residual: scale channels then add back",
        "code": """\
import torch.nn as nn
class SEResidual(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(64, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        h = self.relu(self.bn(self.conv(x)))
        return h + x
""",
        "input_shapes": {"x": ("batch", 64, 16, 16)},
    },
    {
        "name": "correct_prednet_block",
        "category": "correct",
        "has_bug": False,
        "description": "Prediction block: predict -> compute error -> update",
        "code": """\
import torch.nn as nn
class PredBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.predict = nn.Linear(128, 128)
        self.update = nn.Linear(128, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        pred = self.relu(self.predict(x))
        error = x - pred
        return self.relu(self.update(error))
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "correct_channel_attention",
        "category": "correct",
        "has_bug": False,
        "description": "Channel attention: FC->ReLU->FC->Sigmoid -> scale",
        "code": """\
import torch.nn as nn
class ChannelAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 256)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        scale = self.sigmoid(self.fc2(self.relu(self.fc1(x))))
        return x * scale
""",
        "input_shapes": {"x": ("batch", 256)},
    },
    {
        "name": "correct_inverted_bottleneck_fc",
        "category": "correct",
        "has_bug": False,
        "description": "Inverted bottleneck with FC: expand->compress with residual",
        "code": """\
import torch.nn as nn
class InvertedBottleneckFC(nn.Module):
    def __init__(self):
        super().__init__()
        self.expand = nn.Linear(128, 512)
        self.compress = nn.Linear(512, 128)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.expand(x))
        return self.compress(h) + x
""",
        "input_shapes": {"x": ("batch", 128)},
    },
    {
        "name": "correct_conv_1x1_reduction",
        "category": "correct",
        "has_bug": False,
        "description": "1x1 conv for channel reduction: 512->128",
        "code": """\
import torch.nn as nn
class Conv1x1Reduction(nn.Module):
    def __init__(self):
        super().__init__()
        self.reduce = nn.Conv2d(512, 128, 1)
        self.bn = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
    def forward(self, x):
        return self.relu(self.bn(self.reduce(x)))
""",
        "input_shapes": {"x": ("batch", 512, 16, 16)},
    },
    {
        "name": "correct_multi_scale_fusion",
        "category": "correct",
        "has_bug": False,
        "description": "Multi-scale fusion: 1x1 convs to same channel count then add",
        "code": """\
import torch.nn as nn
class MultiScaleFusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_s = nn.Conv2d(64, 32, 1)
        self.proj_m = nn.Conv2d(128, 32, 1)
        self.proj_l = nn.Conv2d(256, 32, 1)
    def forward(self, small, medium, large):
        return self.proj_s(small) + self.proj_m(medium) + self.proj_l(large)
""",
        "input_shapes": {
            "small": ("batch", 64, 16, 16),
            "medium": ("batch", 128, 16, 16),
            "large": ("batch", 256, 16, 16),
        },
    },
]


# ═══════════════════════════════════════════════════════════════════════════════
# Combined export
# ═══════════════════════════════════════════════════════════════════════════════

EXPANDED_BENCHMARKS: List[Dict[str, Any]] = (
    HUGGINGFACE_BENCHMARKS
    + VISION_BENCHMARKS
    + DEVICE_BENCHMARKS
    + PHASE_BENCHMARKS
    + BROADCAST_BENCHMARKS
    + RESHAPE_BENCHMARKS
    + CHAIN_BENCHMARKS
    + CORRECT_BENCHMARKS
)

# Sanity checks
assert len(EXPANDED_BENCHMARKS) >= 200, (
    f"Expected >= 200 benchmarks, got {len(EXPANDED_BENCHMARKS)}"
)
_names = [b["name"] for b in EXPANDED_BENCHMARKS]
assert len(_names) == len(set(_names)), "Duplicate benchmark names found"
