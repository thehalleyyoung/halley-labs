"""Convolutional extensions for finite-width NTK analysis.

Extends the kernel engine and corrections modules to handle convolutional
architectures via patch Gram matrices, translation-equivariant kernel
structure, and Kronecker-factored finite-width corrections.
"""

from .conv_kernel import ConvConfig, ConvKernelResult, ConvNTKComputer
from .patch_gram import PatchGramConfig, PatchGramMatrix
from .conv_corrections import (
    ConvCorrectionConfig,
    ConvCorrectionResult,
    ConvFiniteWidthCorrector,
)
from .cnn_ntk import (
    CNN,
    CNNConfig,
    compute_cnn_h_tensor,
    cnn_ntk_correction_from_h,
)
