"""
Architecture-Specific Kernels for neural network phase diagram analysis.

Implements kernels for attention mechanisms, normalization layers, and
pooling operations, extending the NTK framework to modern architectures.
"""

from .attention_kernel import (
    SoftmaxAttentionKernel,
    MultiHeadAttentionKernel,
    AttentionPatternAnalyzer,
    SelfAttentionRecursion,
    PositionEncodingKernel,
)
from .normalization_kernel import (
    BatchNormKernel,
    LayerNormKernel,
    GroupNormKernel,
    NormalizationRegularizer,
    NormalizationMeanField,
)
from .pooling_kernel import (
    MaxPoolingKernel,
    AveragePoolingKernel,
    GlobalAveragePoolingKernel,
    AdaptivePoolingKernel,
    PoolingSpatialAnalyzer,
)
from .transformer_kernel import (
    TransformerNTKConfig,
    SelfAttentionNTK,
    MultiHeadAttentionNTK,
    LayerNormKernelEffect,
    PositionalEncodingKernel,
    TransformerFiniteWidthCorrections,
    TransformerPhaseBoundary,
)
from .resnet_kernel import (
    ResNetNTKConfig,
    SkipConnectionKernel,
    BatchNormResNetKernel,
    PreActivationResNet,
    SignalPropagationResNet,
    ResNetFiniteWidthCorrections,
    WidthDepthPhaseDiagram,
)

__all__ = [
    "SoftmaxAttentionKernel",
    "MultiHeadAttentionKernel",
    "AttentionPatternAnalyzer",
    "SelfAttentionRecursion",
    "PositionEncodingKernel",
    "BatchNormKernel",
    "LayerNormKernel",
    "GroupNormKernel",
    "NormalizationRegularizer",
    "NormalizationMeanField",
    "MaxPoolingKernel",
    "AveragePoolingKernel",
    "GlobalAveragePoolingKernel",
    "AdaptivePoolingKernel",
    "PoolingSpatialAnalyzer",
    "TransformerNTKConfig",
    "SelfAttentionNTK",
    "MultiHeadAttentionNTK",
    "LayerNormKernelEffect",
    "PositionalEncodingKernel",
    "TransformerFiniteWidthCorrections",
    "TransformerPhaseBoundary",
    "ResNetNTKConfig",
    "SkipConnectionKernel",
    "BatchNormResNetKernel",
    "PreActivationResNet",
    "SignalPropagationResNet",
    "ResNetFiniteWidthCorrections",
    "WidthDepthPhaseDiagram",
]
