"""Residual architecture support for kernel computation and NTK analysis.

Provides skip-connection kernel propagation, ResNet-specific NTK computation,
gradient flow analysis through residual paths, and depth-dependent kernel
convergence utilities for finite-width phase diagram analysis.
"""

from .skip_connections import (
    SkipType,
    SkipConfig,
    SkipConnectionHandler,
)
from .resnet_kernel import (
    ResNetBlockConfig,
    ResNetConfig,
    ResNetKernelResult,
    ResNetNTKComputer,
)
