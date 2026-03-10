"""
usability_oracle.pipeline — End-to-end analysis pipeline orchestration.

Provides :class:`PipelineRunner` for orchestrating all stages from parsing
through repair, with caching, parallel execution, and configuration.
"""

from usability_oracle.pipeline.runner import PipelineRunner, PipelineResult, StageResult
from usability_oracle.pipeline.config import (
    FullPipelineConfig,
    StageConfig,
)
from usability_oracle.pipeline.stages import (
    StageExecutor,
    StageRegistry,
)
from usability_oracle.pipeline.cache import ResultCache
from usability_oracle.pipeline.parallel import ParallelExecutor

__all__ = [
    "PipelineRunner",
    "PipelineResult",
    "StageResult",
    "FullPipelineConfig",
    "StageConfig",
    "StageExecutor",
    "StageRegistry",
    "ResultCache",
    "ParallelExecutor",
]
