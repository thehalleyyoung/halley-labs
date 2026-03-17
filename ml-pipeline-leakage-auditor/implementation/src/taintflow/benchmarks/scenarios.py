"""
taintflow.benchmarks.scenarios – Predefined benchmark scenarios.

Provides self-contained leakage scenarios for evaluating TaintFlow's
detection accuracy and bound tightness.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class PipelineStep:
    """A single operation in a simulated benchmark pipeline.

    Attributes:
        name: Human-readable step name.
        op_type: Operation type identifier.
        params: Configuration parameters for the step.
        introduces_leakage: Whether this step introduces leakage.
        leakage_bits: Ground-truth leakage in bits (0 if clean).
    """

    name: str = ""
    op_type: str = "UNKNOWN"
    params: dict[str, Any] = field(default_factory=dict)
    introduces_leakage: bool = False
    leakage_bits: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "name": self.name,
            "op_type": self.op_type,
            "params": self.params,
            "introduces_leakage": self.introduces_leakage,
            "leakage_bits": self.leakage_bits,
        }


@dataclass
class SimulatedPipeline:
    """A sequence of pipeline steps for benchmark evaluation.

    Attributes:
        name: Pipeline identifier.
        steps: Ordered list of pipeline steps.
        n_samples: Number of samples in the synthetic dataset.
        n_features: Number of features.
        test_fraction: Fraction of data used as test set.
    """

    name: str = ""
    steps: list[PipelineStep] = field(default_factory=list)
    n_samples: int = 1000
    n_features: int = 10
    test_fraction: float = 0.2

    def total_leakage_bits(self) -> float:
        """Sum of ground-truth leakage across all steps."""
        return sum(s.leakage_bits for s in self.steps)

    def leaky_steps(self) -> list[PipelineStep]:
        """Return only the steps that introduce leakage."""
        return [s for s in self.steps if s.introduces_leakage]


@dataclass
class BenchmarkScenario:
    """A named, self-contained benchmark scenario.

    Attributes:
        scenario_id: Unique identifier.
        description: Human-readable description.
        pipeline: The simulated pipeline.
        expected_severity: Expected overall severity classification.
        tags: Classification tags (e.g., 'scaling', 'encoding', 'cv').
    """

    scenario_id: str = ""
    description: str = ""
    pipeline: SimulatedPipeline = field(default_factory=SimulatedPipeline)
    expected_severity: str = "NEGLIGIBLE"
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "scenario_id": self.scenario_id,
            "description": self.description,
            "expected_severity": self.expected_severity,
            "tags": self.tags,
        }


class ScenarioValidator:
    """Validates benchmark scenarios for internal consistency.

    Checks that ground-truth annotations match pipeline structure,
    leakage bits are non-negative, and required fields are populated.
    """

    def validate(self, scenario: BenchmarkScenario) -> list[str]:
        """Validate a scenario and return a list of error messages.

        Returns:
            Empty list if valid, otherwise list of error descriptions.
        """
        errors: list[str] = []
        if not scenario.scenario_id:
            errors.append("scenario_id must be non-empty")
        if not scenario.pipeline.steps:
            errors.append("Pipeline must have at least one step")
        for step in scenario.pipeline.steps:
            if step.leakage_bits < 0:
                errors.append(f"Step '{step.name}' has negative leakage_bits")
        return errors


class StandardBenchmarks:
    """Library of predefined benchmark scenarios covering common leakage types.

    Provides factory methods for standard scenarios used in TaintFlow's
    evaluation suite.
    """

    @staticmethod
    def scaling_before_split() -> BenchmarkScenario:
        """StandardScaler applied to full dataset before train/test split."""
        return BenchmarkScenario(
            scenario_id="scaling_before_split",
            description="StandardScaler.fit_transform on full dataset before split",
            pipeline=SimulatedPipeline(
                name="scaling_leak",
                steps=[
                    PipelineStep(name="load_data", op_type="DATA_SOURCE"),
                    PipelineStep(
                        name="scale", op_type="STANDARD_SCALER",
                        introduces_leakage=True, leakage_bits=0.5,
                    ),
                    PipelineStep(name="split", op_type="TRAIN_TEST_SPLIT"),
                    PipelineStep(name="train", op_type="FIT"),
                ],
            ),
            expected_severity="WARNING",
            tags=["scaling", "preprocessing"],
        )

    @staticmethod
    def target_encoding_leak() -> BenchmarkScenario:
        """Target encoding computed on full dataset."""
        return BenchmarkScenario(
            scenario_id="target_encoding_leak",
            description="Target encoding using full dataset targets",
            pipeline=SimulatedPipeline(
                name="target_enc_leak",
                steps=[
                    PipelineStep(name="load_data", op_type="DATA_SOURCE"),
                    PipelineStep(
                        name="target_encode", op_type="TARGET_ENCODER",
                        introduces_leakage=True, leakage_bits=3.2,
                    ),
                    PipelineStep(name="split", op_type="TRAIN_TEST_SPLIT"),
                    PipelineStep(name="train", op_type="FIT"),
                ],
            ),
            expected_severity="CRITICAL",
            tags=["encoding", "target_leakage"],
        )

    @staticmethod
    def clean_pipeline() -> BenchmarkScenario:
        """Correctly structured pipeline with no leakage."""
        return BenchmarkScenario(
            scenario_id="clean_pipeline",
            description="sklearn Pipeline with split-before-preprocess",
            pipeline=SimulatedPipeline(
                name="clean",
                steps=[
                    PipelineStep(name="load_data", op_type="DATA_SOURCE"),
                    PipelineStep(name="split", op_type="TRAIN_TEST_SPLIT"),
                    PipelineStep(name="scale", op_type="STANDARD_SCALER"),
                    PipelineStep(name="train", op_type="FIT"),
                ],
            ),
            expected_severity="NEGLIGIBLE",
            tags=["clean", "baseline"],
        )

    @classmethod
    def all_scenarios(cls) -> list[BenchmarkScenario]:
        """Return all predefined benchmark scenarios."""
        return [
            cls.scaling_before_split(),
            cls.target_encoding_leak(),
            cls.clean_pipeline(),
        ]


__all__ = [
    "BenchmarkScenario",
    "PipelineStep",
    "ScenarioValidator",
    "SimulatedPipeline",
    "StandardBenchmarks",
]
