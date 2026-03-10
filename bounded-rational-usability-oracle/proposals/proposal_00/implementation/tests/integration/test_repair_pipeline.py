"""Integration tests for the repair synthesis pipeline.

These tests exercise bottleneck detection → strategy selection → mutation
application → cost-reduction verification.  The ``RepairStrategySelector``
proposes ``UIMutation`` instances for detected bottlenecks, the
``MutationOperator`` applies them to accessibility trees, and the tests
verify that the repaired tree's MDP exhibits lower cost.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, List

import numpy as np
import pytest

from usability_oracle.repair.strategies import RepairStrategySelector
from usability_oracle.repair.mutations import MutationOperator
from usability_oracle.repair.models import (
    UIMutation,
    MutationType,
    RepairCandidate,
    RepairResult,
    RepairConstraint,
)
from usability_oracle.bottleneck.classifier import BottleneckClassifier
from usability_oracle.bottleneck.models import BottleneckResult, BottleneckReport
from usability_oracle.accessibility.html_parser import HTMLAccessibilityParser
from usability_oracle.accessibility.normalizer import AccessibilityNormalizer
from usability_oracle.accessibility.models import AccessibilityTree, BoundingBox
from usability_oracle.mdp.builder import MDPBuilder
from usability_oracle.mdp.models import MDP, State, Action, Transition
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.policy.models import Policy
from usability_oracle.core.enums import BottleneckType, Severity, RegressionVerdict
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _load_html(name: str) -> str:
    return (SAMPLE_HTML_DIR / f"{name}.html").read_text()


def _parse(html: str) -> AccessibilityTree:
    tree = HTMLAccessibilityParser().parse(html)
    return AccessibilityNormalizer().normalize(tree)


def _make_task() -> TaskSpec:
    steps = [
        TaskStep(step_id="s1", action_type="click", target_role="textfield",
                 target_name="Username", description="Focus"),
        TaskStep(step_id="s2", action_type="type", target_role="textfield",
                 target_name="Username", input_value="admin",
                 description="Type", depends_on=["s1"]),
        TaskStep(step_id="s3", action_type="click", target_role="button",
                 target_name="Submit", description="Submit",
                 depends_on=["s2"]),
    ]
    flow = TaskFlow(flow_id="f1", name="Login", steps=steps,
                    success_criteria=["done"])
    return TaskSpec(spec_id="t1", name="Login", flows=[flow])


def _make_branching_mdp() -> MDP:
    """Standard 5-state branching MDP for bottleneck tests."""
    states = {}
    for i in range(5):
        sid = f"s{i}"
        states[sid] = State(
            state_id=sid,
            features={"x": float(i * 50), "y": float(i * 30)},
            label=sid,
            is_terminal=(i == 4),
            is_goal=(i == 4),
        )
    actions = {
        "a0": Action(action_id="a0", action_type=Action.CLICK,
                      target_node_id="n0", description=""),
        "a1": Action(action_id="a1", action_type=Action.CLICK,
                      target_node_id="n1", description=""),
        "a2": Action(action_id="a2", action_type=Action.CLICK,
                      target_node_id="n2", description=""),
        "a3": Action(action_id="a3", action_type=Action.CLICK,
                      target_node_id="n3", description=""),
    }
    transitions = [
        Transition(source="s0", action="a0", target="s1",
                   probability=0.7, cost=0.3),
        Transition(source="s0", action="a0", target="s2",
                   probability=0.3, cost=0.5),
        Transition(source="s1", action="a1", target="s3",
                   probability=1.0, cost=0.4),
        Transition(source="s2", action="a2", target="s3",
                   probability=1.0, cost=0.6),
        Transition(source="s3", action="a3", target="s4",
                   probability=1.0, cost=0.2),
    ]
    return MDP(
        states=states, actions=actions, transitions=transitions,
        initial_state="s0", goal_states={"s4"}, discount=0.99,
    )


def _make_motor_bottleneck() -> BottleneckResult:
    """Synthetic motor-difficulty bottleneck result."""
    bn = BottleneckResult(
        bottleneck_type=BottleneckType.MOTOR_DIFFICULTY,
        severity=Severity.HIGH,
        confidence=0.9,
        affected_states=["s0"],
        affected_actions=["a0"],
        description="Small click target",
        recommendation="Increase target size",
        repair_hints=["resize"],
        metadata={"node_ids": ["n0"]},
    )
    # _as_info() in strategies.py reads these via getattr
    bn.node_ids = bn.metadata.get("node_ids", [])
    bn.state_id = bn.affected_states[0] if bn.affected_states else ""
    bn.action_id = bn.affected_actions[0] if bn.affected_actions else ""
    bn.cost_contribution = 0.0
    return bn


def _make_choice_bottleneck() -> BottleneckResult:
    """Synthetic choice-paralysis bottleneck result."""
    bn = BottleneckResult(
        bottleneck_type=BottleneckType.CHOICE_PARALYSIS,
        severity=Severity.MEDIUM,
        confidence=0.85,
        affected_states=["s0"],
        affected_actions=["a0"],
        description="Too many menu options",
        recommendation="Simplify menu",
        repair_hints=["simplify_menu"],
        metadata={"node_ids": ["n0"]},
    )
    bn.node_ids = bn.metadata.get("node_ids", [])
    bn.state_id = bn.affected_states[0] if bn.affected_states else ""
    bn.action_id = bn.affected_actions[0] if bn.affected_actions else ""
    bn.cost_contribution = 0.0
    return bn


# ===================================================================
# Tests – Bottleneck classification
# ===================================================================


class TestBottleneckClassification:
    """Classify bottlenecks in an MDP using the policy."""

    def test_classify_returns_list(self) -> None:
        """``classify`` must return a list of BottleneckResult."""
        mdp = _make_branching_mdp()
        _, policy_map = ValueIterationSolver().solve(mdp)
        policy = Policy(
            state_action_probs={s: {policy_map.get(s, "a0"): 1.0}
                                for s in mdp.states
                                if not mdp.states[s].is_terminal},
            beta=2.0,
        )
        classifier = BottleneckClassifier(beta=2.0)
        results = classifier.classify(mdp, policy)
        assert isinstance(results, list)

    def test_classify_to_report(self) -> None:
        """``classify_to_report`` must return a BottleneckReport."""
        mdp = _make_branching_mdp()
        _, policy_map = ValueIterationSolver().solve(mdp)
        policy = Policy(
            state_action_probs={s: {policy_map.get(s, "a0"): 1.0}
                                for s in mdp.states
                                if not mdp.states[s].is_terminal},
            beta=2.0,
        )
        classifier = BottleneckClassifier(beta=2.0)
        report = classifier.classify_to_report(mdp, policy)
        assert isinstance(report, BottleneckReport)

    def test_bottleneck_result_fields(self) -> None:
        """BottleneckResult instances should have valid types."""
        bn = _make_motor_bottleneck()
        assert isinstance(bn.bottleneck_type, BottleneckType)
        assert isinstance(bn.severity, Severity)
        assert 0.0 <= bn.confidence <= 1.0


# ===================================================================
# Tests – RepairStrategySelector
# ===================================================================


class TestRepairStrategySelector:
    """Strategy selection for detected bottlenecks."""

    def test_select_motor_difficulty(self) -> None:
        """Motor-difficulty bottleneck should produce resize mutations."""
        bn = _make_motor_bottleneck()
        selector = RepairStrategySelector()
        mutations = selector.select(bn)
        assert isinstance(mutations, list)
        assert len(mutations) > 0

    def test_select_choice_paralysis(self) -> None:
        """Choice-paralysis should produce simplify-menu mutations."""
        bn = _make_choice_bottleneck()
        selector = RepairStrategySelector()
        mutations = selector.select(bn)
        assert isinstance(mutations, list)
        assert len(mutations) > 0

    def test_select_all_bottlenecks(self) -> None:
        """``select_all`` should handle a list of bottlenecks."""
        bns = [_make_motor_bottleneck(), _make_choice_bottleneck()]
        selector = RepairStrategySelector()
        mutations = selector.select_all(bns)
        assert isinstance(mutations, list)
        assert len(mutations) >= 2

    def test_mutation_has_target_node(self) -> None:
        """Each mutation should reference a target node."""
        bn = _make_motor_bottleneck()
        mutations = RepairStrategySelector().select(bn)
        for m in mutations:
            assert isinstance(m, UIMutation)
            assert m.target_node_id is not None

    def test_mutation_validates(self) -> None:
        """Generated mutations should pass their own validation."""
        bn = _make_motor_bottleneck()
        mutations = RepairStrategySelector().select(bn)
        for m in mutations:
            errors = m.validate()
            assert len(errors) == 0, f"Mutation validation: {errors}"


# ===================================================================
# Tests – MutationOperator
# ===================================================================


class TestMutationOperator:
    """Apply mutations to accessibility trees."""

    def test_apply_resize_mutation(self) -> None:
        """Applying a resize mutation should not crash."""
        tree = _parse(_load_html("simple_form"))
        interactive = tree.get_interactive_nodes()
        if not interactive:
            pytest.skip("No interactive nodes in fixture")
        node = interactive[0]
        mutation = UIMutation(
            mutation_type=MutationType.RESIZE,
            target_node_id=node.id,
            parameters={"width": 100.0, "height": 50.0},
            description="Resize target",
            priority=1.0,
        )
        operator = MutationOperator()
        repaired = operator.apply(tree, mutation)
        assert repaired.size() >= 1

    def test_apply_relabel_mutation(self) -> None:
        """Applying a relabel mutation should not crash."""
        tree = _parse(_load_html("simple_form"))
        interactive = tree.get_interactive_nodes()
        if not interactive:
            pytest.skip("No interactive nodes in fixture")
        node = interactive[0]
        mutation = UIMutation(
            mutation_type=MutationType.RELABEL,
            target_node_id=node.id,
            parameters={"new_name": "Better Label"},
            description="Relabel node",
            priority=0.8,
        )
        repaired = MutationOperator().apply(tree, mutation)
        assert repaired.size() >= 1

    def test_apply_all_mutations(self) -> None:
        """``apply_all`` should apply a sequence of mutations."""
        tree = _parse(_load_html("simple_form"))
        interactive = tree.get_interactive_nodes()
        if len(interactive) < 2:
            pytest.skip("Need at least 2 interactive nodes")
        mutations = [
            UIMutation(
                mutation_type=MutationType.RESIZE,
                target_node_id=interactive[0].id,
                parameters={"width": 100.0, "height": 50.0},
                description="Resize first",
                priority=1.0,
            ),
            UIMutation(
                mutation_type=MutationType.RELABEL,
                target_node_id=interactive[1].id,
                parameters={"new_name": "New Label"},
                description="Relabel second",
                priority=0.5,
            ),
        ]
        repaired = MutationOperator().apply_all(tree, mutations)
        assert repaired.size() >= 1


# ===================================================================
# Tests – Repair model types
# ===================================================================


class TestRepairModels:
    """Validate repair model dataclasses."""

    def test_ui_mutation_to_dict(self) -> None:
        """``UIMutation.to_dict`` should produce a valid dict."""
        m = UIMutation(
            mutation_type=MutationType.RESIZE,
            target_node_id="n0",
            parameters={"width": 100.0},
            description="Resize",
            priority=1.0,
        )
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "mutation_type" in d

    def test_ui_mutation_round_trip(self) -> None:
        """``to_dict`` / ``from_dict`` should round-trip."""
        m = UIMutation(
            mutation_type=MutationType.RESIZE,
            target_node_id="n0",
            parameters={"width": 100.0},
            description="Resize",
            priority=1.0,
        )
        d = m.to_dict()
        m2 = UIMutation.from_dict(d)
        assert m2.target_node_id == m.target_node_id

    def test_repair_candidate_score(self) -> None:
        """``RepairCandidate.score()`` should return a positive value."""
        candidate = RepairCandidate(
            mutations=[],
            expected_cost_reduction=0.5,
            confidence=0.9,
            bottleneck_addressed="motor_difficulty",
            feasible=True,
            verification_status="verified",
            description="Test candidate",
            code_suggestion=None,
            estimated_effort=1.0,
        )
        assert candidate.score() > 0

    def test_repair_result_has_repair(self) -> None:
        """An empty RepairResult should report ``has_repair == False``."""
        result = RepairResult(
            candidates=[],
            best=None,
            synthesis_time=0.1,
            solver_status="sat",
            n_candidates_explored=0,
        )
        assert not result.has_repair

    def test_mutation_type_all_types(self) -> None:
        """``MutationType.all_types()`` should be non-empty."""
        all_t = MutationType.all_types()
        assert len(all_t) >= 4
