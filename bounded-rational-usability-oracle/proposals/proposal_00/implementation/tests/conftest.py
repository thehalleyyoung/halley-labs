"""Shared pytest fixtures for the Bounded-Rational Usability Oracle test suite.

Provides reusable accessibility trees, MDPs, task specs, cost elements,
policies, and configuration objects used across unit, integration, and
property-based tests.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------
from usability_oracle.core.types import (
    BoundingBox,
    CostTuple,
    Interval,
    Point2D,
    Trajectory,
    TrajectoryStep,
)
from usability_oracle.core.enums import (
    AccessibilityRole,
    BottleneckType,
    CognitiveLaw,
    ComparisonMode,
    EditOperationType,
    MotorChannel,
    OutputFormat,
    PerceptualChannel,
    PipelineStage,
    RegressionVerdict,
    Severity,
)
from usability_oracle.core.config import (
    AlignmentConfig,
    BisimulationConfig,
    CognitiveConfig,
    ComparisonConfig,
    MDPConfig,
    OracleConfig,
    OutputConfig,
    ParserConfig,
    PipelineConfig,
    PolicyConfig,
    RepairConfig,
)
from usability_oracle.core.errors import (
    UsabilityOracleError,
    ParseError,
    AlignmentError,
    CostModelError,
)

# ---------------------------------------------------------------------------
# Domain modules
# ---------------------------------------------------------------------------
from usability_oracle.interval import Interval as IvInterval
from usability_oracle.algebra.models import CostElement, Leaf, Sequential, Parallel
from usability_oracle.mdp.models import State, Action, Transition, MDP
from usability_oracle.policy.models import Policy, QValues
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec
from usability_oracle.bisimulation.models import Partition

# ---------------------------------------------------------------------------
# Fixtures directory
# ---------------------------------------------------------------------------
FIXTURES_DIR = Path(__file__).parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"
SAMPLE_JSON_DIR = FIXTURES_DIR / "sample_json"


# ===================================================================
# Interval fixtures
# ===================================================================

@pytest.fixture
def unit_interval() -> IvInterval:
    """The canonical [0, 1] interval."""
    return IvInterval(0.0, 1.0)


@pytest.fixture
def positive_interval() -> IvInterval:
    """Strictly positive interval [1, 5]."""
    return IvInterval(1.0, 5.0)


@pytest.fixture
def negative_interval() -> IvInterval:
    """Strictly negative interval [-5, -1]."""
    return IvInterval(-5.0, -1.0)


@pytest.fixture
def symmetric_interval() -> IvInterval:
    """Symmetric interval [-2, 2]."""
    return IvInterval(-2.0, 2.0)


@pytest.fixture
def degenerate_interval() -> IvInterval:
    """A degenerate (point) interval [3, 3]."""
    return IvInterval(3.0, 3.0)


# ===================================================================
# Point / BoundingBox fixtures
# ===================================================================

@pytest.fixture
def origin_point() -> Point2D:
    return Point2D(x=0.0, y=0.0)


@pytest.fixture
def sample_point() -> Point2D:
    return Point2D(x=100.0, y=200.0)


@pytest.fixture
def small_bbox() -> BoundingBox:
    """A 24×24 minimum-touch-target bounding box."""
    return BoundingBox(x=10.0, y=10.0, width=24.0, height=24.0)


@pytest.fixture
def medium_bbox() -> BoundingBox:
    """A 100×40 typical button bounding box."""
    return BoundingBox(x=50.0, y=100.0, width=100.0, height=40.0)


@pytest.fixture
def large_bbox() -> BoundingBox:
    """A 400×300 dashboard-panel bounding box."""
    return BoundingBox(x=0.0, y=0.0, width=400.0, height=300.0)


@pytest.fixture
def viewport_bbox() -> BoundingBox:
    """A typical 1920×1080 viewport."""
    return BoundingBox(x=0.0, y=0.0, width=1920.0, height=1080.0)


# ===================================================================
# CostElement / CostTuple fixtures
# ===================================================================

@pytest.fixture
def zero_cost() -> CostElement:
    return CostElement.zero()


@pytest.fixture
def small_cost() -> CostElement:
    """A small cognitive cost (≈ 200 ms mean, low variance)."""
    return CostElement(mu=0.200, sigma_sq=0.001, kappa=0.0, lambda_=0.0)


@pytest.fixture
def medium_cost() -> CostElement:
    """A medium cognitive cost (≈ 800 ms mean)."""
    return CostElement(mu=0.800, sigma_sq=0.010, kappa=0.01, lambda_=0.0)


@pytest.fixture
def large_cost() -> CostElement:
    """A large cognitive cost (≈ 2 s mean, high variance)."""
    return CostElement(mu=2.0, sigma_sq=0.100, kappa=0.05, lambda_=0.01)


@pytest.fixture
def cost_elements() -> List[CostElement]:
    """A list of diverse cost elements for composition tests."""
    return [
        CostElement(mu=0.1, sigma_sq=0.001),
        CostElement(mu=0.3, sigma_sq=0.005, kappa=0.01),
        CostElement(mu=0.5, sigma_sq=0.010, kappa=0.02, lambda_=0.005),
        CostElement(mu=1.0, sigma_sq=0.050, kappa=0.05),
    ]


# ===================================================================
# MDP fixtures
# ===================================================================

def _make_simple_mdp() -> MDP:
    """Create a minimal 3-state deterministic MDP for testing.

    States: start -> middle -> goal
    Actions: click (start->middle), type (middle->goal)
    """
    states = {
        "start": State(
            state_id="start",
            features={"x": 0.0, "y": 0.0},
            label="start",
            is_terminal=False,
            is_goal=False,
            metadata={},
        ),
        "middle": State(
            state_id="middle",
            features={"x": 100.0, "y": 50.0},
            label="middle",
            is_terminal=False,
            is_goal=False,
            metadata={},
        ),
        "goal": State(
            state_id="goal",
            features={"x": 200.0, "y": 100.0},
            label="goal",
            is_terminal=True,
            is_goal=True,
            metadata={},
        ),
    }
    actions = {
        "click_button": Action(
            action_id="click_button",
            action_type=Action.CLICK,
            target_node_id="btn1",
            description="Click the button",
            preconditions=[],
        ),
        "type_text": Action(
            action_id="type_text",
            action_type=Action.TYPE,
            target_node_id="input1",
            description="Type into text field",
            preconditions=[],
        ),
    }
    transitions = [
        Transition(source="start", action="click_button", target="middle", probability=1.0, cost=0.5),
        Transition(source="middle", action="type_text", target="goal", probability=1.0, cost=1.0),
    ]
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="start",
        goal_states={"goal"},
        discount=0.99,
    )


def _make_branching_mdp() -> MDP:
    """Create a 5-state branching MDP with stochastic transitions.

    States: s0 → s1 / s2 → s3 → goal
    """
    states = {}
    for i in range(5):
        sid = f"s{i}"
        states[sid] = State(
            state_id=sid,
            features={"x": float(i * 50), "y": float(i * 30)},
            label=sid,
            is_terminal=(i == 4),
            is_goal=(i == 4),
            metadata={},
        )
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK, target_node_id="n0", description="", preconditions=[]),
        "a1": Action(action_id="a1", action_type=Action.CLICK, target_node_id="n1", description="", preconditions=[]),
        "a2": Action(action_id="a2", action_type=Action.CLICK, target_node_id="n2", description="", preconditions=[]),
        "a3": Action(action_id="a3", action_type=Action.CLICK, target_node_id="n3", description="", preconditions=[]),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1", probability=0.7, cost=0.3),
        Transition(source="s0", action="a0", target="s2", probability=0.3, cost=0.5),
        Transition(source="s1", action="a1", target="s3", probability=1.0, cost=0.4),
        Transition(source="s2", action="a2", target="s3", probability=1.0, cost=0.6),
        Transition(source="s3", action="a3", target="s4", probability=1.0, cost=0.2),
    ]
    return MDP(
        states=states,
        actions=actions,
        transitions=transitions,
        initial_state="s0",
        goal_states={"s4"},
        discount=0.99,
    )


@pytest.fixture
def simple_mdp() -> MDP:
    """Minimal 3-state linear MDP."""
    return _make_simple_mdp()


@pytest.fixture
def branching_mdp() -> MDP:
    """5-state branching MDP with stochastic transitions."""
    return _make_branching_mdp()


# ===================================================================
# Policy fixtures
# ===================================================================

@pytest.fixture
def deterministic_policy() -> Policy:
    """A deterministic policy for the simple MDP."""
    return Policy(
        state_action_probs={
            "start": {"click_button": 1.0},
            "middle": {"type_text": 1.0},
        },
        beta=10.0,
        values={"start": 1.5, "middle": 1.0, "goal": 0.0},
        q_values={
            "start": {"click_button": 1.5},
            "middle": {"type_text": 1.0},
        },
        metadata={},
    )


@pytest.fixture
def uniform_policy() -> Policy:
    """A uniform random policy for branching MDP."""
    return Policy(
        state_action_probs={
            "s0": {"a0": 1.0},
            "s1": {"a1": 1.0},
            "s2": {"a2": 1.0},
            "s3": {"a3": 1.0},
        },
        beta=1.0,
        values={"s0": 1.5, "s1": 0.6, "s2": 0.8, "s3": 0.2, "s4": 0.0},
        q_values={
            "s0": {"a0": 1.5},
            "s1": {"a1": 0.6},
            "s2": {"a2": 0.8},
            "s3": {"a3": 0.2},
        },
        metadata={},
    )


# ===================================================================
# TaskSpec fixtures
# ===================================================================

@pytest.fixture
def simple_task_spec() -> TaskSpec:
    """A simple form-filling task spec."""
    steps = [
        TaskStep(
            step_id="step_1",
            action_type="click",
            target_role="textfield",
            target_name="Username",
            description="Click username field",
        ),
        TaskStep(
            step_id="step_2",
            action_type="type",
            target_role="textfield",
            target_name="Username",
            input_value="testuser",
            description="Type username",
            depends_on=["step_1"],
        ),
        TaskStep(
            step_id="step_3",
            action_type="click",
            target_role="button",
            target_name="Submit",
            description="Click submit",
            depends_on=["step_2"],
        ),
    ]
    flow = TaskFlow(
        flow_id="login_flow",
        name="Login",
        steps=steps,
        success_criteria=["form_submitted"],
        description="Log in to the application",
    )
    return TaskSpec(
        spec_id="login_task",
        name="Login Task",
        description="Complete the login form",
        flows=[flow],
    )


@pytest.fixture
def navigation_task_spec() -> TaskSpec:
    """A navigation task spec."""
    steps = [
        TaskStep(step_id="nav_1", action_type="click", target_role="link",
                 target_name="Products", description="Click products link"),
        TaskStep(step_id="nav_2", action_type="click", target_role="link",
                 target_name="Category A", description="Select category",
                 depends_on=["nav_1"]),
        TaskStep(step_id="nav_3", action_type="click", target_role="button",
                 target_name="Add to Cart", description="Add item",
                 depends_on=["nav_2"]),
    ]
    flow = TaskFlow(flow_id="nav_flow", name="Navigate", steps=steps)
    return TaskSpec(spec_id="nav_task", name="Navigation Task", flows=[flow])


# ===================================================================
# Partition fixtures
# ===================================================================

@pytest.fixture
def trivial_partition() -> Partition:
    """All states in one block."""
    return Partition.trivial(["s0", "s1", "s2", "s3", "s4"])


@pytest.fixture
def discrete_partition() -> Partition:
    """Each state in its own block."""
    return Partition.discrete(["s0", "s1", "s2", "s3", "s4"])


# ===================================================================
# Configuration fixtures
# ===================================================================

@pytest.fixture
def default_oracle_config() -> OracleConfig:
    return OracleConfig.default()


@pytest.fixture
def cognitive_config() -> CognitiveConfig:
    return CognitiveConfig()


@pytest.fixture
def mdp_config() -> MDPConfig:
    return MDPConfig()


@pytest.fixture
def comparison_config() -> ComparisonConfig:
    return ComparisonConfig()


# ===================================================================
# Sample HTML/JSON loading fixtures
# ===================================================================

@pytest.fixture
def simple_form_html() -> str:
    """Load the simple form HTML fixture."""
    p = SAMPLE_HTML_DIR / "simple_form.html"
    return p.read_text() if p.exists() else ""


@pytest.fixture
def navigation_menu_html() -> str:
    p = SAMPLE_HTML_DIR / "navigation_menu.html"
    return p.read_text() if p.exists() else ""


@pytest.fixture
def complex_dashboard_html() -> str:
    p = SAMPLE_HTML_DIR / "complex_dashboard.html"
    return p.read_text() if p.exists() else ""


@pytest.fixture
def modal_dialog_html() -> str:
    p = SAMPLE_HTML_DIR / "modal_dialog.html"
    return p.read_text() if p.exists() else ""


@pytest.fixture
def simple_form_json() -> str:
    p = SAMPLE_JSON_DIR / "simple_form.json"
    return p.read_text() if p.exists() else ""


@pytest.fixture
def navigation_menu_json() -> str:
    p = SAMPLE_JSON_DIR / "navigation_menu.json"
    return p.read_text() if p.exists() else ""


# ===================================================================
# Numpy random generator fixture
# ===================================================================

@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic random generator for reproducible tests."""
    return np.random.default_rng(42)
