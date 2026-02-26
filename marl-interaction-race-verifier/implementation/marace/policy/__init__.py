"""
Policy ingestion and analysis module for the MARACE system.

Provides ONNX model loading, Lipschitz bound extraction, abstract policy
evaluation over zonotope inputs (DeepZ / CROWN), and policy utility classes
for testing and benchmarking.
"""

from marace.policy.onnx_loader import (
    ActivationType,
    InputOutputSpec,
    LayerExtractor,
    LayerInfo,
    ModelLoader,
    ModelValidator,
    NetworkArchitecture,
    ONNXPolicy,
    PolicyEvaluator,
)
from marace.policy.lipschitz import (
    LayerLipschitz,
    LipschitzCertificate,
    LipschitzExtractor,
    LocalLipschitz,
    ReLULipschitz,
    SpectralNormComputation,
    TanhLipschitz,
)
from marace.policy.abstract_policy import (
    AbstractOutput,
    AbstractPolicyEvaluator,
    BacksubstitutionRefiner,
    BatchNormAbstractTransformer,
    DeepZTransformer,
    LinearAbstractTransformer,
    PrecisionTracker,
    ReLUAbstractTransformer,
    TanhAbstractTransformer,
)
from marace.policy.policy_utils import (
    DummyPolicy,
    LinearPolicy,
    NormalizationWrapper,
    PolicyCache,
    PolicyComparator,
    PolicySampler,
    RandomPolicy,
)
from marace.policy.lipsdp import (
    SpectralNormProductBound,
    LipSDPBound,
    LocalLipschitzBound,
    RecursiveBound,
    LipschitzTightnessAnalysis,
    CascadingErrorAnalysis,
)

__all__ = [
    # onnx_loader
    "ActivationType",
    "InputOutputSpec",
    "LayerExtractor",
    "LayerInfo",
    "ModelLoader",
    "ModelValidator",
    "NetworkArchitecture",
    "ONNXPolicy",
    "PolicyEvaluator",
    # lipschitz
    "LayerLipschitz",
    "LipschitzCertificate",
    "LipschitzExtractor",
    "LocalLipschitz",
    "ReLULipschitz",
    "SpectralNormComputation",
    "TanhLipschitz",
    # abstract_policy
    "AbstractOutput",
    "AbstractPolicyEvaluator",
    "BacksubstitutionRefiner",
    "BatchNormAbstractTransformer",
    "DeepZTransformer",
    "LinearAbstractTransformer",
    "PrecisionTracker",
    "ReLUAbstractTransformer",
    "TanhAbstractTransformer",
    # policy_utils
    "DummyPolicy",
    "LinearPolicy",
    "NormalizationWrapper",
    "PolicyCache",
    "PolicyComparator",
    "PolicySampler",
    "RandomPolicy",
    "SpectralNormProductBound",
    "LipSDPBound",
    "LocalLipschitzBound",
    "RecursiveBound",
    "LipschitzTightnessAnalysis",
    "CascadingErrorAnalysis",
]
