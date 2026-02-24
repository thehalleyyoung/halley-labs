"""Architecture Intermediate Representation (IR) for neural network analysis.

Provides types, computation graph nodes, graph construction/manipulation,
and parsing utilities for converting neural network architectures into
an analyzable IR suitable for NTK and phase diagram computation.
"""

from .types import (
    LayerType,
    ActivationType,
    NormalizationType,
    InitializationType,
    TensorShape,
    ScalingExponents,
    KernelRecursionType,
)
from .nodes import (
    AbstractNode,
    DenseNode,
    Conv1DNode,
    Conv2DNode,
    ActivationNode,
    NormNode,
    ResidualNode,
    PoolingNode,
    FlattenNode,
    DropoutNode,
    InputNode,
    OutputNode,
)
from .graph import ComputationGraph
from .parser import ArchitectureParser
