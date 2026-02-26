"""Tests for pipeline orchestration."""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from marace.pipeline import MARACEPipeline, PipelineConfig, PipelineState, PipelineStage


class TestPipelineConfig:
    """Test pipeline configuration."""

    def test_default_config(self):
        """Test default configuration with required args."""
        config = PipelineConfig(
            env_config={"type": "highway"},
            policy_paths=["policy0.onnx"],
        )
        assert config.num_trace_samples > 0
        assert config.importance_samples > 0

    def test_custom_config(self):
        """Test custom configuration."""
        config = PipelineConfig(
            env_config={"type": "highway"},
            policy_paths=["p0.onnx", "p1.onnx"],
            adversarial_budget=200,
            importance_samples=5000,
        )
        assert config.adversarial_budget == 200
        assert config.importance_samples == 5000


class TestPipelineState:
    """Test pipeline state tracking."""

    def test_initial_state(self):
        """Test initial pipeline state."""
        state = PipelineState()
        assert state.current_stage == PipelineStage.LOAD_POLICIES

    def test_state_transitions(self):
        """Test state transitions via mark_stage_complete."""
        state = PipelineState()
        state.mark_stage_complete(PipelineStage.LOAD_POLICIES)
        assert PipelineStage.LOAD_POLICIES in state.completed_stages

    def test_completed_stages_tracked(self):
        """Test completed stages list grows."""
        state = PipelineState()
        state.mark_stage_complete(PipelineStage.LOAD_POLICIES)
        state.mark_stage_complete(PipelineStage.CONFIGURE_ENV)
        state.mark_stage_complete(PipelineStage.PARSE_SPEC)
        assert len(state.completed_stages) >= 3


class TestMARACEPipeline:
    """Test MARACE pipeline."""

    def test_pipeline_creation(self):
        """Test creating pipeline."""
        config = PipelineConfig(
            env_config={"type": "highway", "num_agents": 2},
            policy_paths=["p0.onnx"],
        )
        pipeline = MARACEPipeline(config=config)
        assert pipeline is not None

    def test_pipeline_has_run_and_summary(self):
        """Test pipeline has run and summary methods."""
        config = PipelineConfig(
            env_config={"type": "highway"},
            policy_paths=["p0.onnx"],
        )
        pipeline = MARACEPipeline(config=config)
        assert hasattr(pipeline, "run")
        assert hasattr(pipeline, "summary")

    def test_pipeline_summary(self):
        """Test pipeline summary method."""
        config = PipelineConfig(
            env_config={"type": "highway"},
            policy_paths=["p0.onnx"],
        )
        pipeline = MARACEPipeline(config=config)
        summary = pipeline.summary()
        assert summary is not None
