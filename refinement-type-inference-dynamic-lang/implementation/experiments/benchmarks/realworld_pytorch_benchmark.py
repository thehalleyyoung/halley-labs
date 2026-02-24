"""
Real-world PyTorch nn.Module benchmarks for LiquidPy evaluation.

Benchmarks modeled after real patterns from:
- Common PyTorch GitHub issues about shape mismatches
- StackOverflow questions about RuntimeError: shape mismatch
- Bug-fix commits in open-source PyTorch projects
- Real architectures (ResNet, BERT, GPT, U-Net, DCGAN, etc.)
"""

REALWORLD_PYTORCH_BENCHMARKS = {

    # =========================================================================
    # BUGGY MODELS (~30)
    # =========================================================================

    # --- shape_mismatch (Linear dimension) ---

    "resnet_fc_after_pool_bug": {
        "source": """
class ResNetFCBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, 10)  # BUG: should be 64, not 128

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "ResNet-style flatten-to-FC dimension mismatch — common in GitHub Issues when changing backbone width",
        "bug_description": "nn.Linear expects 128 features but AdaptiveAvgPool2d(1,1) over 64 channels yields 64 features",
    },

    "vgg_classifier_mismatch": {
        "source": """
class VGGClassifierBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 4096),  # BUG: should be 128*56*56 only for 224 input
            nn.ReLU(),
            nn.Linear(4096, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "VGG classifier dimension bug — StackOverflow pattern: trained on ImageNet 224x224, deployed on CIFAR 32x32",
        "bug_description": "Linear expects 128*56*56=401408 features but 32x32 input after two MaxPool2d(2) yields 128*8*8=8192",
    },

    "bert_pooler_dim_bug": {
        "source": """
class BERTPoolerBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(512, 768)
        self.layernorm = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.pooler_dense = nn.Linear(512, 768)  # BUG: should be 768
        self.pooler_act = nn.Tanh()

    def forward(self, x):
        x = self.layernorm(self.embedding(x))
        x, _ = self.attention(x, x, x)
        cls_token = x[:, 0, :]
        return self.pooler_act(self.pooler_dense(cls_token))
""",
        "input_shapes": {"x": ("batch", 128, 512)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "BERT pooler dense layer input dimension bug — GitHub PR pattern where hidden_size was changed but pooler not updated",
        "bug_description": "pooler_dense expects 512 features but cls_token has 768 features from attention output",
    },

    "mlp_hidden_chain_bug": {
        "source": """
class MLPChainBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(128, 64)  # BUG: input should be 256 not 128
        self.fc3 = nn.Linear(64, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Classic MLP layer chain mismatch — most common StackOverflow beginner error",
        "bug_description": "fc2 expects 128 features but fc1 outputs 256 features",
    },

    "gpt_lm_head_dim_bug": {
        "source": """
class GPTLMHeadBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50257, 768)
        self.ln = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        self.lm_head = nn.Linear(512, 50257)  # BUG: should be 768

    def forward(self, x):
        x = self.embed(x)
        x = self.ln(x)
        attn_out, _ = self.attention(x, x, x)
        x = x + attn_out
        x = x + self.ffn(x)
        return self.lm_head(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "GPT language model head dimension mismatch — pattern from nanoGPT Issues where d_model was changed",
        "bug_description": "lm_head expects 512 features but transformer outputs 768-dim embeddings",
    },

    "autoencoder_bottleneck_bug": {
        "source": """
class AutoencoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),  # BUG: should be 16 not 32
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Autoencoder bottleneck mismatch — common in VAE tutorials when latent_dim is edited",
        "bug_description": "Decoder first layer expects 32 features but encoder bottleneck outputs 16",
    },

    # --- conv channel bugs ---

    "resnet_skip_channel_bug": {
        "source": """
class ResNetSkipBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU()
        # BUG: no projection for skip connection, 64 != 128

    def forward(self, x):
        identity = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(identity)))
        return out + identity  # shape mismatch: 128 vs 64 channels
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "ResNet skip connection channel mismatch — GitHub Issue #5305 pattern: forgetting 1x1 projection",
        "bug_description": "Skip connection adds 128-channel output to 64-channel identity without projection",
    },

    "unet_encoder_decoder_channel_bug": {
        "source": """
class UNetDecoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(64, 64, 3, padding=1)  # BUG: should be 128 (64 + 64 concat)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        d1 = self.up(e2)
        # skip connection concatenation makes channels 64+64=128
        d1 = torch.cat([d1, e1], dim=1)
        return self.relu(self.dec1(d1))
""",
        "input_shapes": {"x": ("batch", 1, 64, 64)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "U-Net decoder skip connection channel mismatch — common mistake in medical imaging repos",
        "bug_description": "dec1 expects 64 input channels but cat of upsampled (64) + skip (64) = 128 channels",
    },

    "dcgan_generator_conv_bug": {
        "source": """
class DCGANGeneratorBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(100, 512 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # BUG: in_channels should be 256
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.project(x)
        x = x.view(x.size(0), 512, 4, 4)
        return self.main(x)
""",
        "input_shapes": {"x": ("batch", 100)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "DCGAN generator channel mismatch — pattern from PyTorch DCGAN tutorial Issues",
        "bug_description": "Second ConvTranspose2d expects 128 input channels but first outputs 256 channels",
    },

    "mobilenet_depthwise_bug": {
        "source": """
class DepthwiseSepBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.bn_dw = nn.BatchNorm2d(32)
        self.pw = nn.Conv2d(64, 64, 1)  # BUG: in_channels should be 32
        self.bn_pw = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "MobileNet-style depthwise separable conv channel mismatch — common when editing channel multiplier",
        "bug_description": "Pointwise conv expects 64 input channels but depthwise conv outputs 32 channels",
    },

    "conv_batchnorm_mismatch": {
        "source": """
class ConvBNMismatch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)  # BUG: should be 32
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Conv-BatchNorm channel mismatch — high-frequency bug in PyTorch GitHub Issues",
        "bug_description": "BatchNorm2d(64) expects 64 channels but conv1 outputs 32 channels",
    },

    # --- reshape/view bugs ---

    "dcgan_reshape_bug": {
        "source": """
class DCGANReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(100, 256 * 4 * 4)
        self.deconv1 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.bn1 = nn.BatchNorm2d(128)
        self.deconv2 = nn.ConvTranspose2d(128, 3, 4, 2, 1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = x.view(x.size(0), 512, 2, 2)  # BUG: 512*2*2=2048 != 256*4*4=4096
        x = self.relu(self.bn1(self.deconv1(x)))
        return self.tanh(self.deconv2(x))
""",
        "input_shapes": {"x": ("batch", 100)},
        "is_buggy": True,
        "category": "reshape",
        "source_description": "DCGAN generator reshape bug — StackOverflow pattern: wrong view dimensions after projection",
        "bug_description": "view reshapes to 512*2*2=2048 but fc outputs 256*4*4=4096 elements",
    },

    "vit_patch_reshape_bug": {
        "source": """
class ViTPatchBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.ln = nn.LayerNorm(768)
        self.attention = nn.MultiheadAttention(768, 12, batch_first=True)
        self.fc = nn.Linear(768, 10)

    def forward(self, x):
        x = self.patch_embed(x)  # (B, 768, 14, 14) for 224x224
        x = x.view(x.size(0), 768, -1)  # (B, 768, 196)
        # BUG: need to transpose to (B, 196, 768) for attention
        x = self.ln(x)
        x, _ = self.attention(x, x, x)
        return self.fc(x[:, 0, :])
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "reshape",
        "source_description": "Vision Transformer patch embedding reshape bug — missing transpose after conv, from ViT reimplementation Issues",
        "bug_description": "LayerNorm(768) applied to last dim of (B,768,196) which is 196 not 768; missing permute",
    },

    "flatten_wrong_start_dim": {
        "source": """
class FlattenStartDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((2, 2)),
        )
        self.flatten = nn.Flatten(start_dim=0)  # BUG: should be start_dim=1
        self.classifier = nn.Linear(256, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "reshape",
        "source_description": "Flatten start_dim=0 bug — flattens batch dimension, common beginner mistake on StackOverflow",
        "bug_description": "Flatten(start_dim=0) collapses batch dimension, producing 1D tensor instead of (batch, 256)",
    },

    "transformer_seq_reshape_bug": {
        "source": """
class TransformerReshapeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(512, 256)
        self.attention = nn.MultiheadAttention(256, 8, batch_first=True)
        self.ln = nn.LayerNorm(256)
        self.head = nn.Linear(256, 100)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attention(x, x, x)
        x = self.ln(x)
        x = x.view(x.size(0), -1)  # BUG: flattens seq*256, then Linear(256,100) won't match
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 50, 512)},
        "is_buggy": True,
        "category": "reshape",
        "source_description": "Transformer output reshape bug — flattening sequence dimension before classification head",
        "bug_description": "view flattens to (batch, 50*256=12800) but head expects (batch, 256)",
    },

    # --- broadcasting bugs ---

    "attention_scale_broadcast_bug": {
        "source": """
class AttentionScaleBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(256, 256)
        self.k_proj = nn.Linear(256, 256)
        self.v_proj = nn.Linear(256, 256)
        self.scale = nn.Linear(256, 1)  # BUG: scale should be scalar, not per-token
        self.out_proj = nn.Linear(256, 256)

    def forward(self, x):
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        s = self.scale(q)  # (B, seq, 1) — mismatched with attention logits
        attn = torch.matmul(q, k.transpose(-2, -1)) * s  # broadcast issue
        attn = torch.softmax(attn, dim=-1)
        return self.out_proj(torch.matmul(attn, v))
""",
        "input_shapes": {"x": ("batch", 32, 256)},
        "is_buggy": True,
        "category": "broadcast",
        "source_description": "Attention scaling broadcast bug — learned scale creates wrong broadcasting with attention matrix",
        "bug_description": "scale output (B,seq,1) broadcasts incorrectly with attn logits (B,seq,seq) — should be scalar",
    },

    "residual_broadcast_dim_bug": {
        "source": """
class ResidualBroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        # No downsample projection for spatial dims

    def forward(self, x):
        identity = self.relu(self.bn1(self.conv1(x)))  # (B, 64, H, W)
        out = self.relu(self.bn2(self.conv2(identity)))  # (B, 64, H/2, W/2)
        return out + identity  # BUG: spatial dims don't match
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "broadcast",
        "source_description": "ResNet residual spatial dimension mismatch — stride-2 conv without matching downsample on skip",
        "bug_description": "identity is (B,64,32,32) but out is (B,64,16,16) — spatial dims mismatch in addition",
    },

    "feature_pyramid_broadcast_bug": {
        "source": """
class FPNBroadcastBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_low = nn.Conv2d(3, 64, 3, padding=1)
        self.backbone_high = nn.Sequential(
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.lateral = nn.Conv2d(128, 64, 1)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        low = self.relu(self.backbone_low(x))   # (B, 64, H, W)
        high = self.backbone_high(low)            # (B, 128, H/2, W/2)
        high_lat = self.lateral(high)             # (B, 64, H/2, W/2)
        fused = low + high_lat  # BUG: spatial dims differ
        return self.fc(self.pool(fused).view(x.size(0), -1))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "broadcast",
        "source_description": "Feature Pyramid Network lateral fusion without upsampling — missing F.interpolate",
        "bug_description": "low is (B,64,32,32) but high_lat is (B,64,16,16) — spatial size mismatch in addition",
    },

    # --- phase-dependent bugs ---

    "dropout_eval_shape_bug": {
        "source": """
class DropoutPhaseBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)  # BUG: should be 256
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": True,
        "category": "phase",
        "source_description": "Dropout does not change shape but developer confused about output dim — StackOverflow pattern",
        "bug_description": "fc2 expects 512 features but dropout preserves fc1 output size of 256",
    },

    "batchnorm_eval_mismatch": {
        "source": """
class BNEvalBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)  # BUG: should be 64
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "phase",
        "source_description": "BatchNorm2d channel mismatch after conv — manifests in both train and eval but often caught late in eval",
        "bug_description": "bn2 expects 32 channels but conv2 outputs 64 channels",
    },

    # --- More shape_mismatch ---

    "multihead_attention_embed_dim_bug": {
        "source": """
class MHAEmbedBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(100, 256)
        self.ln = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)  # BUG: embed_dim=512 != 256
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.ln(self.embed(x))
        x, _ = self.attn(x, x, x)
        return self.fc(x[:, 0, :])
""",
        "input_shapes": {"x": ("batch", 20, 100)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "MultiheadAttention embed_dim mismatch — common when changing hidden size in transformer configs",
        "bug_description": "MultiheadAttention expects 512-dim input but LayerNorm outputs 256-dim",
    },

    "layernorm_dim_bug": {
        "source": """
class LayerNormDimBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 512)
        self.ln = nn.LayerNorm(768)  # BUG: should be 512
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.fc = nn.Linear(512, 10)

    def forward(self, x):
        x = self.embed(x)
        x = self.ln(x)
        x, _ = self.attn(x, x, x)
        return self.fc(x.mean(dim=1))
""",
        "input_shapes": {"x": ("batch", 64)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "LayerNorm normalized_shape mismatch — from HuggingFace Transformers Issues",
        "bug_description": "LayerNorm(768) expects last dim=768 but embedding outputs dim=512",
    },

    "siamese_projection_bug": {
        "source": """
class SiameseProjectionBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(128, 256),  # BUG: backbone outputs 64 not 128
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "SimCLR/BYOL projection head dim mismatch — common in self-supervised learning repos",
        "bug_description": "Projector expects 128 features but backbone with 64 filters + AdaptiveAvgPool2d(1) outputs 64",
    },

    "seq2seq_decoder_bug": {
        "source": """
class Seq2SeqDecoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed = nn.Embedding(5000, 256)
        self.encoder_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.decoder_embed = nn.Embedding(5000, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.fc = nn.Linear(512, 5000)  # BUG: should be 256
        self.ln = nn.LayerNorm(256)

    def forward(self, x):
        enc = self.encoder_embed(x)
        enc, _ = self.encoder_attn(enc, enc, enc)
        dec = self.decoder_embed(x)
        dec, _ = self.cross_attn(dec, enc, enc)
        dec = self.ln(dec)
        return self.fc(dec)
""",
        "input_shapes": {"x": ("batch", 50)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Seq2Seq decoder output projection dim bug — pattern from fairseq Issues",
        "bug_description": "fc expects 512 features but cross-attention output is 256-dim",
    },

    "classification_head_after_attention_bug": {
        "source": """
class ClassificationHeadBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Linear(300, 512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln = nn.LayerNorm(512)
        self.pool = nn.AdaptiveAvgPool2d(1)  # BUG: using 2d pool on 3d tensor
        self.fc = nn.Linear(512, 5)

    def forward(self, x):
        x = self.embed(x)
        x, _ = self.attn(x, x, x)
        x = self.ln(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 40, 300)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Using AdaptiveAvgPool2d on 3D tensor — common confusion between sequence and image pooling",
        "bug_description": "AdaptiveAvgPool2d expects 4D input but attention output is 3D (batch, seq, dim)",
    },

    "gan_discriminator_fc_bug": {
        "source": """
class GANDiscriminatorBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256 * 8 * 8, 1)  # BUG: 64/2/2/2=8 but actually 4x4 for 32x32

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "GAN discriminator flatten dimension calculation error — common when changing image size",
        "bug_description": "classifier expects 256*8*8=16384 but three stride-2 convs on 32x32 yield 256*4*4=4096",
    },

    "embedding_layernorm_bug": {
        "source": """
class EmbeddingLNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embed = nn.Embedding(30000, 768)
        self.pos_embed = nn.Embedding(512, 384)  # BUG: should be 768
        self.ln = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.fc = nn.Linear(768, 30000)

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        tok = self.token_embed(x)
        pos = self.pos_embed(positions)
        x = tok + pos  # BUG: 768 + 384 can't add
        x = self.ln(x)
        x, _ = self.attn(x, x, x)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 128)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Positional embedding dimension mismatch with token embedding — GPT/BERT config error pattern",
        "bug_description": "pos_embed outputs 384-dim but token_embed outputs 768-dim — cannot add",
    },

    "convnext_block_bug": {
        "source": """
class ConvNeXtBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(96, 96, 7, padding=3, groups=96)
        self.norm = nn.LayerNorm(96)
        self.pwconv1 = nn.Linear(96, 384)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(384, 48)  # BUG: should be 96 for residual
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(96, 10)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)
        x = x + identity  # BUG: 48 channels != 96 channels
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 96, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "ConvNeXt block residual dimension mismatch — pwconv2 output != input channels",
        "bug_description": "pwconv2 outputs 48-dim but residual connection needs 96 to match identity",
    },

    "efficientnet_se_bug": {
        "source": """
class SEBlockBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 32)  # BUG: should be 64
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        se = self.se_pool(x).view(x.size(0), -1)
        se = self.sigmoid(self.se_fc2(self.relu(self.se_fc1(se))))
        se = se.view(x.size(0), -1, 1, 1)
        x = x * se  # BUG: se has 32 channels, x has 64
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "broadcast",
        "source_description": "Squeeze-and-Excitation block channel mismatch — se_fc2 outputs wrong channel count",
        "bug_description": "SE branch outputs 32 channels but feature map has 64 channels — broadcast mismatch in multiply",
    },

    "detr_query_dim_bug": {
        "source": """
class DETRQueryBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_conv = nn.Conv2d(3, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.input_proj = nn.Linear(256, 256)
        self.query_embed = nn.Embedding(100, 512)  # BUG: should be 256
        self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.fc = nn.Linear(256, 91)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.backbone_conv(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 256)
        x = self.input_proj(x)
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.cross_attn(queries, x, x)  # BUG: queries 512-dim != 256
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "DETR object query dimension mismatch — query embedding dim != transformer hidden dim",
        "bug_description": "query_embed has 512-dim but cross_attn expects 256-dim queries",
    },

    "swin_patch_merge_bug": {
        "source": """
class SwinPatchMergeBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 96, 4, stride=4)
        self.bn1 = nn.BatchNorm2d(96)
        self.conv2 = nn.Conv2d(96, 192, 2, stride=2)
        self.bn2 = nn.BatchNorm2d(192)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(96, 10)  # BUG: should be 192

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Swin Transformer head dimension mismatch — head not updated after adding patch merging stage",
        "bug_description": "head expects 96 features but last stage outputs 192 channels after pooling",
    },

    "dual_encoder_concat_bug": {
        "source": """
class DualEncoderBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_encoder = nn.Sequential(
            nn.Linear(300, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, 64),  # image outputs 64-dim
        )
        self.classifier = nn.Linear(256, 10)  # BUG: concat is 128+64=192 not 256

    def forward(self, x):
        # Simulate: use same input for both modalities for shape checking
        text_feat = self.text_encoder(x.view(x.size(0), -1)[:, :300])
        img_feat = self.image_encoder(x[:, :3, :, :])
        fused = torch.cat([text_feat, img_feat], dim=1)
        return self.classifier(fused)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "Multi-modal fusion concatenation dimension error — text+image feature concat wrong total",
        "bug_description": "classifier expects 256 but concat of text(128) + image(64) = 192",
    },

    "wav2vec_feature_proj_bug": {
        "source": """
class Wav2VecProjBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(80, 512),
            nn.GELU(),
            nn.Linear(512, 768),
        )
        self.ln = nn.LayerNorm(768)
        self.projection = nn.Linear(768, 256)
        self.classifier = nn.Linear(768, 32)  # BUG: should be 256

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.ln(x)
        x = self.projection(x)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 100, 80)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "wav2vec 2.0 feature projection → classifier dim mismatch — projection changes dim but classifier not updated",
        "bug_description": "classifier expects 768 but projection reduces to 256",
    },

    "gpt_ffn_inner_dim_bug": {
        "source": """
class GPTFFNBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(50000, 512)
        self.ln1 = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(1024, 512),  # BUG: should be 2048 input
        )
        self.head = nn.Linear(512, 50000)

    def forward(self, x):
        x = self.embed(x)
        x2 = self.ln1(x)
        a, _ = self.attn(x2, x2, x2)
        x = x + a
        x2 = self.ln2(x)
        x = x + self.ffn(x2)
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "GPT feed-forward network inner dimension mismatch — second Linear input != first Linear output",
        "bug_description": "ffn second layer expects 1024 but first layer outputs 2048",
    },

    "unet_skip_spatial_bug": {
        "source": """
class UNetSpatialBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 32, 3, padding=1)
        self.enc2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.bottleneck = nn.Conv2d(64, 128, 3, padding=1)
        # BUG: ConvTranspose2d with wrong stride yields wrong spatial size
        self.up = nn.ConvTranspose2d(128, 64, 3, stride=3, padding=1)
        self.dec = nn.Conv2d(128, 64, 3, padding=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        b = self.relu(self.bottleneck(self.pool(e2)))
        d = self.up(b)
        # Skip concat — spatial dims won't match with stride=3
        d = torch.cat([d, e2], dim=1)
        return self.relu(self.dec(d))
""",
        "input_shapes": {"x": ("batch", 1, 64, 64)},
        "is_buggy": True,
        "category": "reshape",
        "source_description": "U-Net upsampling spatial size mismatch — ConvTranspose2d stride doesn't match MaxPool stride",
        "bug_description": "ConvTranspose2d stride=3 doesn't recover MaxPool2d stride=2 spatial dims, skip connection cat fails",
    },

    "cifar_pool_flatten_bug": {
        "source": """
class CIFARPoolBug(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        # After 2 pools on 32x32: 32/4=8, so 32*8*8=2048
        self.fc1 = nn.Linear(32 * 4 * 4, 128)  # BUG: should be 32*8*8 (only 1 pool in forward)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        # Developer forgot second pool
        x = x.view(x.size(0), -1)
        return self.fc2(self.relu(self.fc1(x)))
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": True,
        "category": "shape_mismatch",
        "source_description": "CIFAR tutorial model — forgot a pooling layer but computed FC dim assuming two pools",
        "bug_description": "fc1 expects 32*4*4=512 but only one pool applied: actual is 32*16*16=8192",
    },

    # =========================================================================
    # CORRECT MODELS (~20)
    # =========================================================================

    "simple_cnn_correct": {
        "source": """
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc = nn.Linear(64 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Standard CIFAR-10 CNN — correct baseline",
        "bug_description": "correct",
    },

    "resnet_block_correct": {
        "source": """
class ResNetBlockCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        identity = x
        out = self.relu(self.bn2(self.conv2(x)))
        out = out + identity
        out = self.pool(out).view(out.size(0), -1)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "ResNet basic block with matching channels — correct skip connection",
        "bug_description": "correct",
    },

    "resnet_downsample_correct": {
        "source": """
class ResNetDownsampleCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        identity = self.downsample(x)
        out = self.relu(self.bn2(self.conv2(x)))
        out = out + identity
        out = self.pool(out).view(out.size(0), -1)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "ResNet block with correct 1x1 projection downsample",
        "bug_description": "correct",
    },

    "mlp_correct": {
        "source": """
class MLPCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        return self.fc3(x)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Standard MNIST MLP with dropout — correct baseline",
        "bug_description": "correct",
    },

    "transformer_encoder_correct": {
        "source": """
class TransformerEncoderCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(10000, 512)
        self.ln1 = nn.LayerNorm(512)
        self.attn = nn.MultiheadAttention(512, 8, batch_first=True)
        self.ln2 = nn.LayerNorm(512)
        self.ffn = nn.Sequential(
            nn.Linear(512, 2048),
            nn.GELU(),
            nn.Linear(2048, 512),
        )
        self.head = nn.Linear(512, 10000)

    def forward(self, x):
        x = self.embed(x)
        x2 = self.ln1(x)
        a, _ = self.attn(x2, x2, x2)
        x = x + a
        x2 = self.ln2(x)
        x = x + self.ffn(x2)
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 64)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Standard transformer encoder block with pre-norm — correct",
        "bug_description": "correct",
    },

    "dcgan_generator_correct": {
        "source": """
class DCGANGeneratorCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Linear(100, 256 * 4 * 4)
        self.main = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 3, 4, 2, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.project(x)
        x = x.view(x.size(0), 256, 4, 4)
        return self.main(x)
""",
        "input_shapes": {"x": ("batch", 100)},
        "is_buggy": False,
        "category": "reshape",
        "source_description": "DCGAN generator with correct reshape and channel progression",
        "bug_description": "correct",
    },

    "vgg_style_correct": {
        "source": """
class VGGStyleCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "VGG-style network for CIFAR-10 with correct flatten dimensions",
        "bug_description": "correct",
    },

    "bert_encoder_correct": {
        "source": """
class BERTEncoderCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(30000, 768)
        self.pos_embed = nn.Embedding(512, 768)
        self.ln = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        self.ln2 = nn.LayerNorm(768)
        self.pooler = nn.Linear(768, 768)
        self.tanh = nn.Tanh()

    def forward(self, x):
        positions = torch.arange(x.size(1), device=x.device).unsqueeze(0)
        x = self.ln(self.embed(x) + self.pos_embed(positions))
        a, _ = self.attn(x, x, x)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return self.tanh(self.pooler(x[:, 0, :]))
""",
        "input_shapes": {"x": ("batch", 128)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "BERT encoder with correct embeddings and pooler — correct",
        "bug_description": "correct",
    },

    "autoencoder_correct": {
        "source": """
class AutoencoderCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
        )
        self.decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid(),
        )

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)
""",
        "input_shapes": {"x": ("batch", 784)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Symmetric autoencoder with matching bottleneck — correct",
        "bug_description": "correct",
    },

    "unet_correct": {
        "source": """
class UNetCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = nn.Conv2d(1, 64, 3, padding=1)
        self.enc2 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Conv2d(128, 64, 3, padding=1)
        self.out_conv = nn.Conv2d(64, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        e1 = self.relu(self.enc1(x))
        e2 = self.relu(self.enc2(self.pool(e1)))
        d1 = self.up(e2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.relu(self.dec1(d1))
        return self.out_conv(d1)
""",
        "input_shapes": {"x": ("batch", 1, 64, 64)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "U-Net with correct skip connections and channel handling",
        "bug_description": "correct",
    },

    "mobilenet_correct": {
        "source": """
class MobileNetCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.dw = nn.Conv2d(32, 32, 3, padding=1, groups=32)
        self.bn_dw = nn.BatchNorm2d(32)
        self.pw = nn.Conv2d(32, 64, 1)
        self.bn_pw = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn_dw(self.dw(x)))
        x = self.relu(self.bn_pw(self.pw(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "MobileNet-style depthwise separable block — correct",
        "bug_description": "correct",
    },

    "se_block_correct": {
        "source": """
class SEBlockCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_fc1 = nn.Linear(64, 16)
        self.se_fc2 = nn.Linear(16, 64)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        se = self.se_pool(x).view(x.size(0), -1)
        se = self.sigmoid(self.se_fc2(self.relu(self.se_fc1(se))))
        se = se.view(x.size(0), -1, 1, 1)
        x = x * se
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "broadcast",
        "source_description": "Squeeze-and-Excitation block with correct channel dimensions",
        "bug_description": "correct",
    },

    "gan_discriminator_correct": {
        "source": """
class GANDiscriminatorCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(256 * 4 * 4, 1)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "GAN discriminator with correct flatten dimension for 32x32 input",
        "bug_description": "correct",
    },

    "text_classifier_correct": {
        "source": """
class TextClassifierCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(20000, 256)
        self.ln = nn.LayerNorm(256)
        self.attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(256, 1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )
        self.ln2 = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, 5)

    def forward(self, x):
        x = self.embed(x)
        x = self.ln(x)
        a, _ = self.attn(x, x, x)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return self.classifier(x.mean(dim=1))
""",
        "input_shapes": {"x": ("batch", 100)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Transformer-based text classifier with mean pooling — correct",
        "bug_description": "correct",
    },

    "convnext_block_correct": {
        "source": """
class ConvNeXtBlockCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.dwconv = nn.Conv2d(96, 96, 7, padding=3, groups=96)
        self.norm = nn.LayerNorm(96)
        self.pwconv1 = nn.Linear(96, 384)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(384, 96)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(96, 10)

    def forward(self, x):
        identity = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv2(self.act(self.pwconv1(x)))
        x = x.permute(0, 3, 1, 2)
        x = x + identity
        x = self.pool(x).view(x.size(0), -1)
        return self.head(x)
""",
        "input_shapes": {"x": ("batch", 96, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "ConvNeXt block with correct inverted bottleneck and residual",
        "bug_description": "correct",
    },

    "seq2seq_correct": {
        "source": """
class Seq2SeqCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_embed = nn.Embedding(5000, 256)
        self.encoder_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.decoder_embed = nn.Embedding(5000, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.ln = nn.LayerNorm(256)
        self.fc = nn.Linear(256, 5000)

    def forward(self, x):
        enc = self.encoder_embed(x)
        enc, _ = self.encoder_attn(enc, enc, enc)
        dec = self.decoder_embed(x)
        dec, _ = self.cross_attn(dec, enc, enc)
        dec = self.ln(dec)
        return self.fc(dec)
""",
        "input_shapes": {"x": ("batch", 50)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "Seq2Seq encoder-decoder with correct cross-attention dimensions",
        "bug_description": "correct",
    },

    "siamese_correct": {
        "source": """
class SiameseCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.projector = nn.Sequential(
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )

    def forward(self, x):
        features = self.backbone(x)
        return self.projector(features)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "SimCLR-style siamese network with correct projection head",
        "bug_description": "correct",
    },

    "wav2vec_correct": {
        "source": """
class Wav2VecCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(80, 512),
            nn.GELU(),
            nn.Linear(512, 768),
        )
        self.ln = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        self.classifier = nn.Linear(768, 32)

    def forward(self, x):
        x = self.feature_extractor(x)
        x = self.ln(x)
        a, _ = self.attn(x, x, x)
        x = x + a
        x = x + self.ffn(x)
        return self.classifier(x)
""",
        "input_shapes": {"x": ("batch", 100, 80)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "wav2vec-style speech model with correct feature extraction and classification",
        "bug_description": "correct",
    },

    "vit_classifier_correct": {
        "source": """
class ViTClassifierCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, 768, kernel_size=16, stride=16)
        self.ln1 = nn.LayerNorm(768)
        self.attn = nn.MultiheadAttention(768, 12, batch_first=True)
        self.ln2 = nn.LayerNorm(768)
        self.ffn = nn.Sequential(
            nn.Linear(768, 3072),
            nn.GELU(),
            nn.Linear(3072, 768),
        )
        self.head = nn.Linear(768, 1000)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, 768)
        x2 = self.ln1(x)
        a, _ = self.attn(x2, x2, x2)
        x = x + a
        x = x + self.ffn(self.ln2(x))
        return self.head(x.mean(dim=1))
""",
        "input_shapes": {"x": ("batch", 3, 224, 224)},
        "is_buggy": False,
        "category": "reshape",
        "source_description": "Vision Transformer with correct patch embedding reshape and classification",
        "bug_description": "correct",
    },

    "detr_correct": {
        "source": """
class DETRCorrect(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone_conv = nn.Conv2d(3, 256, 3, padding=1)
        self.bn = nn.BatchNorm2d(256)
        self.pool = nn.AdaptiveAvgPool2d((8, 8))
        self.input_proj = nn.Linear(256, 256)
        self.query_embed = nn.Embedding(100, 256)
        self.cross_attn = nn.MultiheadAttention(256, 8, batch_first=True)
        self.fc = nn.Linear(256, 91)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.bn(self.backbone_conv(x)))
        x = self.pool(x)
        x = x.permute(0, 2, 3, 1).reshape(x.size(0), -1, 256)
        x = self.input_proj(x)
        queries = self.query_embed.weight.unsqueeze(0).expand(x.size(0), -1, -1)
        out, _ = self.cross_attn(queries, x, x)
        return self.fc(out)
""",
        "input_shapes": {"x": ("batch", 3, 64, 64)},
        "is_buggy": False,
        "category": "shape_mismatch",
        "source_description": "DETR-style detector with correct query and feature dimensions",
        "bug_description": "correct",
    },
}


def get_benchmark_summary():
    """Return summary statistics for the benchmark suite."""
    total = len(REALWORLD_PYTORCH_BENCHMARKS)
    buggy = sum(1 for b in REALWORLD_PYTORCH_BENCHMARKS.values() if b["is_buggy"])
    correct = total - buggy

    categories = {}
    buggy_categories = {}
    for name, bench in REALWORLD_PYTORCH_BENCHMARKS.items():
        cat = bench["category"]
        categories[cat] = categories.get(cat, 0) + 1
        if bench["is_buggy"]:
            buggy_categories[cat] = buggy_categories.get(cat, 0) + 1

    return {
        "total_benchmarks": total,
        "buggy_models": buggy,
        "correct_models": correct,
        "categories": categories,
        "buggy_by_category": buggy_categories,
    }


if __name__ == "__main__":
    summary = get_benchmark_summary()
    print(f"Total benchmarks: {summary['total_benchmarks']}")
    print(f"  Buggy:   {summary['buggy_models']}")
    print(f"  Correct: {summary['correct_models']}")
    print("\nBy category (total / buggy):")
    for cat in sorted(summary["categories"]):
        buggy_count = summary["buggy_by_category"].get(cat, 0)
        print(f"  {cat:20s}: {summary['categories'][cat]:3d} total, {buggy_count:3d} buggy")
