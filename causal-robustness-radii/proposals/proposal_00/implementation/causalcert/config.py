"""
Global configuration defaults and environment-variable overrides.

This module provides a singleton-style configuration that is populated from
``PipelineConfig`` but can also be overridden by environment variables for
integration testing and CI.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from causalcert.types import PipelineConfig

# ---------------------------------------------------------------------------
# Environment variable names
# ---------------------------------------------------------------------------

_ENV_PREFIX = "CAUSALCERT_"


@dataclass(slots=True)
class GlobalConfig:
    """Run-time global settings derived from a :class:`PipelineConfig`.

    Attributes
    ----------
    log_level : str
        Logging level (``DEBUG``, ``INFO``, ``WARNING``, …).
    progress_bar : bool
        Whether to display progress bars in the CLI.
    max_threads : int
        Maximum number of threads for parallel work.
    float_precision : str
        Floating point precision for reporting (``"float32"`` or ``"float64"``).
    """

    log_level: str = "INFO"
    progress_bar: bool = True
    max_threads: int = 1
    float_precision: str = "float64"

    @classmethod
    def from_env(cls) -> GlobalConfig:
        """Construct a :class:`GlobalConfig` from environment variables.

        Environment variables follow the pattern ``CAUSALCERT_<FIELD>``
        (upper-case).  Missing variables fall back to defaults.
        """
        return cls(
            log_level=os.environ.get(f"{_ENV_PREFIX}LOG_LEVEL", "INFO"),
            progress_bar=os.environ.get(f"{_ENV_PREFIX}PROGRESS_BAR", "1") == "1",
            max_threads=int(os.environ.get(f"{_ENV_PREFIX}MAX_THREADS", "1")),
            float_precision=os.environ.get(f"{_ENV_PREFIX}FLOAT_PRECISION", "float64"),
        )


def resolve_config(pipeline_cfg: PipelineConfig | None = None) -> GlobalConfig:
    """Merge a :class:`PipelineConfig` with environment overrides.

    Parameters
    ----------
    pipeline_cfg : PipelineConfig | None
        Optional pipeline-level configuration.

    Returns
    -------
    GlobalConfig
    """
    cfg = GlobalConfig.from_env()
    if pipeline_cfg is not None:
        cfg.max_threads = pipeline_cfg.n_jobs if pipeline_cfg.n_jobs > 0 else cfg.max_threads
    return cfg
