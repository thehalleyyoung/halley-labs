"""
CollusionProof: Algorithmic Collusion Certification System.

A framework for detecting, certifying, and analyzing algorithmic collusion
in multi-agent strategic environments using game-theoretic oracle methods.
"""

__version__ = "0.1.0"

# Import from types.py which always exists.
from collusion_proof.types import Verdict, GameConfig, TestResult, CollusionPremiumResult  # noqa: F401

try:
    from collusion_proof.analysis.composite_test import CompositeTest  # noqa: F401
except ImportError:
    CompositeTest = None

try:
    from collusion_proof.oracle.passive_oracle import PassiveOracle  # noqa: F401
except ImportError:
    PassiveOracle = None

try:
    from collusion_proof.oracle.checkpoint_oracle import CheckpointOracle  # noqa: F401
except ImportError:
    CheckpointOracle = None

try:
    from collusion_proof.oracle.rewind_oracle import RewindOracle  # noqa: F401
except ImportError:
    RewindOracle = None

try:
    from collusion_proof.evaluation.benchmark_runner import BenchmarkRunner  # noqa: F401
except ImportError:
    BenchmarkRunner = None

__all__ = [
    "__version__",
    "CompositeTest",
    "GameConfig",
    "TestResult",
    "Verdict",
    "CollusionPremiumResult",
    "PassiveOracle",
    "CheckpointOracle",
    "RewindOracle",
    "BenchmarkRunner",
]
