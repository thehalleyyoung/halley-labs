"""
Benchmark suite of PyTorch nn.Module models for baseline comparison.

Contains models with known shape bugs and correct models for evaluating
TensorGuard vs pyright vs mypy on tensor shape verification.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class BenchmarkModel:
    name: str
    source: str
    has_bug: bool
    bug_description: str  # empty string if no bug


# ---------------------------------------------------------------------------
# BUGGY MODELS (10) — shape mismatches invisible to type checkers
# ---------------------------------------------------------------------------

BUGGY_MODELS: List[BenchmarkModel] = [
    # 1. Linear dimension mismatch
    BenchmarkModel(
        name="linear_dim_mismatch",
        source='''\
import torch
import torch.nn as nn

class LinearDimMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(128, 10)  # Bug: expects 128, gets 256

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x
''',
        has_bug=True,
        bug_description="fc1 outputs 256 but fc2 expects 128",
    ),

    # 2. Conv2d channel mismatch
    BenchmarkModel(
        name="conv_channel_mismatch",
        source='''\
import torch
import torch.nn as nn

class ConvChannelMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 128, kernel_size=3, padding=1)  # Bug: expects 32, gets 64

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
''',
        has_bug=True,
        bug_description="conv1 outputs 64 channels but conv2 expects 32",
    ),

    # 3. BatchNorm feature count mismatch
    BenchmarkModel(
        name="batchnorm_mismatch",
        source='''\
import torch
import torch.nn as nn

class BatchNormMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)  # Bug: expects 32 features, conv outputs 64
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.conv2(x)
        return x
''',
        has_bug=True,
        bug_description="conv1 outputs 64 channels but BatchNorm2d expects 32",
    ),

    # 4. Matmul shape incompatibility
    BenchmarkModel(
        name="matmul_shape_error",
        source='''\
import torch
import torch.nn as nn

class MatmulShapeError(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj_q = nn.Linear(512, 64)
        self.proj_k = nn.Linear(512, 128)  # Bug: key dim 128 != query dim 64

    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        attn = torch.matmul(q, k.transpose(-2, -1))
        return attn
''',
        has_bug=True,
        bug_description="Query dim 64 and key dim 128 are incompatible for attention matmul",
    ),

    # 5. Residual connection shape mismatch
    BenchmarkModel(
        name="residual_shape_mismatch",
        source='''\
import torch
import torch.nn as nn

class ResidualShapeMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        # Bug: no projection for residual — input has 64 channels, output has 128

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + residual  # Shape mismatch: 128 vs 64 channels
        return x
''',
        has_bug=True,
        bug_description="Residual add: x has 128 channels but residual has 64 channels",
    ),

    # 6. LSTM hidden size mismatch with downstream linear
    BenchmarkModel(
        name="lstm_linear_mismatch",
        source='''\
import torch
import torch.nn as nn

class LSTMLinearMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True)
        self.fc = nn.Linear(64, 10)  # Bug: LSTM hidden_size is 128 not 64

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.fc(output[:, -1, :])
        return x
''',
        has_bug=True,
        bug_description="LSTM hidden_size is 128 but fc expects input_features=64",
    ),

    # 7. Flatten then wrong linear input size
    BenchmarkModel(
        name="flatten_linear_mismatch",
        source='''\
import torch
import torch.nn as nn

class FlattenLinearMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)   # 32->28
        self.pool = nn.MaxPool2d(2, 2)                  # 28->14
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)   # 14->10
        # After pool: 32 * 5 * 5 = 800
        self.fc1 = nn.Linear(1024, 120)  # Bug: should be 800 not 1024

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x
''',
        has_bug=True,
        bug_description="After conv+pool the flattened size is 800 but fc1 expects 1024",
    ),

    # 8. Multi-head attention with wrong projection
    BenchmarkModel(
        name="multihead_proj_mismatch",
        source='''\
import torch
import torch.nn as nn

class MultiHeadProjMismatch(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads  # 64
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(256, d_model)  # Bug: should be d_model=512 not 256

    def forward(self, x):
        batch_size = x.size(0)
        q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1))
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(out)
''',
        has_bug=True,
        bug_description="W_o expects 256 but concatenated heads produce 512",
    ),

    # 9. Encoder-decoder dimension mismatch
    BenchmarkModel(
        name="encoder_decoder_mismatch",
        source='''\
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 64)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return self.fc2(x)

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 256)  # Bug: encoder outputs 64 not 32
        self.fc2 = nn.Linear(256, 784)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
''',
        has_bug=True,
        bug_description="Encoder outputs 64-dim but Decoder expects 32-dim input",
    ),

    # 10. Conv transpose output channels feeding wrong conv
    BenchmarkModel(
        name="convtranspose_mismatch",
        source='''\
import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.down = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.up = nn.ConvTranspose2d(128, 32, kernel_size=2, stride=2)
        # Bug: up outputs 32 channels, but skip connection adds 64
        self.conv = nn.Conv2d(96, 64, kernel_size=3, padding=1)
        # 32 + 64 = 96 — looks correct but up should output 64 not 32

    def forward(self, x):
        skip = x  # 64 channels
        x = self.down(x)  # 128 channels
        x = self.up(x)  # 32 channels (bug: should be 64)
        x = torch.cat([x, skip], dim=1)  # 32+64=96
        x = self.conv(x)
        return x
''',
        has_bug=True,
        bug_description="ConvTranspose2d outputs 32 channels; concat with 64-ch skip gives 96 but design intent requires 128",
    ),
]


# ---------------------------------------------------------------------------
# CORRECT MODELS (8) — well-formed, no shape bugs
# ---------------------------------------------------------------------------

CORRECT_MODELS: List[BenchmarkModel] = [
    # 1. Simple MLP
    BenchmarkModel(
        name="correct_mlp",
        source='''\
import torch
import torch.nn as nn

class CorrectMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
''',
        has_bug=False,
        bug_description="",
    ),

    # 2. Simple CNN
    BenchmarkModel(
        name="correct_cnn",
        source='''\
import torch
import torch.nn as nn

class CorrectCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
''',
        has_bug=False,
        bug_description="",
    ),

    # 3. ResNet-style block
    BenchmarkModel(
        name="correct_resblock",
        source='''\
import torch
import torch.nn as nn

class CorrectResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

    def forward(self, x):
        residual = x
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = x + residual
        return torch.relu(x)
''',
        has_bug=False,
        bug_description="",
    ),

    # 4. LSTM classifier
    BenchmarkModel(
        name="correct_lstm",
        source='''\
import torch
import torch.nn as nn

class CorrectLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=128, hidden_size=256, batch_first=True, num_layers=2)
        self.fc = nn.Linear(256, 10)

    def forward(self, x):
        output, (h_n, c_n) = self.lstm(x)
        x = self.fc(output[:, -1, :])
        return x
''',
        has_bug=False,
        bug_description="",
    ),

    # 5. Self-attention block
    BenchmarkModel(
        name="correct_self_attention",
        source='''\
import torch
import torch.nn as nn

class CorrectSelfAttention(nn.Module):
    def __init__(self, d_model=512, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        batch_size = x.size(0)
        q = self.W_q(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.W_k(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.W_v(x).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.n_heads * self.d_k)
        return self.W_o(out)
''',
        has_bug=False,
        bug_description="",
    ),

    # 6. Autoencoder
    BenchmarkModel(
        name="correct_autoencoder",
        source='''\
import torch
import torch.nn as nn

class CorrectAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
''',
        has_bug=False,
        bug_description="",
    ),

    # 7. VGG-style feature extractor
    BenchmarkModel(
        name="correct_vgg_block",
        source='''\
import torch
import torch.nn as nn

class CorrectVGGBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
''',
        has_bug=False,
        bug_description="",
    ),

    # 8. Transformer encoder layer
    BenchmarkModel(
        name="correct_transformer_enc",
        source='''\
import torch
import torch.nn as nn

class CorrectTransformerEnc(nn.Module):
    def __init__(self, d_model=256, dim_ff=1024):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, num_heads=8)
        self.ff = nn.Sequential(
            nn.Linear(d_model, dim_ff),
            nn.ReLU(),
            nn.Linear(dim_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        return x
''',
        has_bug=False,
        bug_description="",
    ),
]

ALL_BENCHMARKS: List[BenchmarkModel] = BUGGY_MODELS + CORRECT_MODELS
