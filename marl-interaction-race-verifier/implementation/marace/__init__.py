"""MARACE: Multi-Agent Race Condition Verifier.

A verification framework for detecting, analyzing, and certifying the absence
of race conditions in multi-agent reinforcement learning (MARL) policies.

The framework combines abstract interpretation, happens-before reasoning,
adversarial schedule search, and importance sampling to provide both
sound verification and probabilistic guarantees.
"""

__version__ = "0.1.0"
__author__ = "MARACE Team"

from typing import Final

PACKAGE_NAME: Final[str] = "marace"

# Lazy imports to avoid circular dependencies and speed up startup
def __getattr__(name: str):  # noqa: N807
    """Lazy-load submodules on first access."""
    _submodules = {
        "abstract",
        "decomposition",
        "env",
        "evaluation",
        "hb",
        "pipeline",
        "policy",
        "race",
        "reporting",
        "sampling",
        "search",
        "spec",
        "trace",
    }
    if name in _submodules:
        import importlib
        return importlib.import_module(f".{name}", __name__)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "PACKAGE_NAME",
    "abstract",
    "decomposition",
    "env",
    "evaluation",
    "hb",
    "pipeline",
    "policy",
    "race",
    "reporting",
    "sampling",
    "search",
    "spec",
    "trace",
]
