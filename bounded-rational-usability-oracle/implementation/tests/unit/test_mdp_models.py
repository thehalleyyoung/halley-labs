"""Unit tests for usability_oracle.mdp.models — State, Action, Transition, MDP.

Validates the core MDP data structures used to represent UI navigation
as a Markov Decision Process: state construction, action validation,
transition semantics, and MDP graph operations (reachability, predecessors,
successors, statistics).
"""

from __future__ import annotations

import pytest

from usability_oracle.mdp.models import State, Action, Transition, MDP, MDPStatistics
from tests.fixtures.sample_mdps import (
    make_two_state_mdp,
    make_cyclic_mdp,
    make_choice_mdp,
    make_large_chain_mdp,
)


# ═══════════════════════════════════════════════════════════════════════════
# State tests
# ═══════════════════════════════════════════════════════════════════════════


class TestState:
    """Tests for the State dataclass — frozen, slotted representation of
    an MDP state encoding the user's focus position and task progress."""

    def test_state_required_fields(self):
        """State must be constructable with only a state_id."""
        s = State(state_id="s0")
        assert s.state_id == "s0"
        assert s.features == {}
        assert s.label == ""
        assert s.is_terminal is False
        assert s.is_goal is False
        assert s.metadata == {}

    def test_state_all_fields(self):
        """State stores all explicitly provided fields."""
        s = State(
            state_id="btn:0b11",
            features={"task_progress": 0.5, "depth": 2.0},
            label="button:Submit @ step 2",
            is_terminal=False,
            is_goal=True,
            metadata={"node_id": "btn", "scroll_offset": 120},
        )
        assert s.state_id == "btn:0b11"
        assert s.features["task_progress"] == 0.5
        assert s.label == "button:Submit @ step 2"
        assert s.is_goal is True
        assert s.metadata["scroll_offset"] == 120

    def test_node_id_from_metadata(self):
        """node_id property should prefer metadata['node_id'] when present."""
        s = State(state_id="x:progress", metadata={"node_id": "input_pw"})
        assert s.node_id == "input_pw"

    def test_node_id_fallback_to_split(self):
        """node_id falls back to the part before ':' in state_id."""
        s = State(state_id="btn_submit:0b10")
        assert s.node_id == "btn_submit"

    def test_node_id_no_colon(self):
        """node_id returns the full state_id when no ':' separator exists."""
        s = State(state_id="simple")
        assert s.node_id == "simple"

    def test_task_progress_from_features(self):
        """task_progress reads from the features dict."""
        s = State(state_id="s", features={"task_progress": 0.75})
        assert s.task_progress == 0.75

    def test_task_progress_default_zero(self):
        """task_progress defaults to 0.0 when absent."""
        s = State(state_id="s")
        assert s.task_progress == 0.0

    def test_working_memory_load(self):
        """working_memory_load property reads from features."""
        s = State(state_id="s", features={"working_memory_load": 3.0})
        assert s.working_memory_load == 3.0

    def test_state_is_frozen(self):
        """State should be immutable (frozen dataclass)."""
        s = State(state_id="s")
        with pytest.raises(AttributeError):
            s.state_id = "other"  # type: ignore[misc]


# ═══════════════════════════════════════════════════════════════════════════
# Action tests
# ═══════════════════════════════════════════════════════════════════════════


class TestAction:
    """Tests for the Action dataclass — an atomic UI interaction with
    type constants and validation logic."""

    def test_action_type_constants(self):
        """All eight action type constants must be defined and accessible on instances."""
        a = Action(action_id="x", action_type="click", target_node_id="t")
        assert a.CLICK == "click"
        assert a.TYPE == "type"
        assert a.TAB == "tab"
        assert a.SCROLL == "scroll"
        assert a.NAVIGATE == "navigate"
        assert a.READ == "read"
        assert a.SELECT == "select"
        assert a.BACK == "back"

    def test_valid_click_action(self):
        """A click action with a target node should validate cleanly."""
        a = Action(action_id="a1", action_type="click", target_node_id="btn1")
        assert a.validate() == []

    def test_validate_missing_action_id(self):
        """Empty action_id should produce a validation error."""
        a = Action(action_id="", action_type="click", target_node_id="btn")
        errors = a.validate()
        assert any("action_id" in e for e in errors)

    def test_validate_unknown_type(self):
        """An unrecognised action_type should be flagged."""
        a = Action(action_id="a1", action_type="swipe", target_node_id="x")
        errors = a.validate()
        assert any("swipe" in e for e in errors)

    def test_validate_click_requires_target(self):
        """Click, type, select, and read actions require a target_node_id."""
        for atype in ("click", "type", "select", "read"):
            a = Action(action_id="a1", action_type=atype, target_node_id=None)
            errors = a.validate()
            assert len(errors) > 0, f"{atype} should require target_node_id"

    def test_validate_global_actions_no_target(self):
        """Tab, scroll, navigate, and back do NOT require a target."""
        for atype in ("tab", "scroll", "navigate", "back"):
            a = Action(action_id="a1", action_type=atype)
            errors = a.validate()
            assert errors == [], f"{atype} should not require target"


# ═══════════════════════════════════════════════════════════════════════════
# Transition tests
# ═══════════════════════════════════════════════════════════════════════════


class TestTransition:
    """Tests for the Transition dataclass — a single probabilistic
    edge T(target | source, action) with cost."""

    def test_valid_transition(self):
        """A well-formed transition has no validation errors."""
        t = Transition(source="s0", action="go", target="s1",
                       probability=0.7, cost=1.5)
        assert t.validate() == []

    def test_probability_out_of_range(self):
        """Probability outside [0, 1] should be invalid."""
        t = Transition(source="s", action="a", target="t",
                       probability=1.5, cost=0.0)
        assert any("probability" in e for e in t.validate())

    def test_negative_probability(self):
        """Negative probability should be invalid."""
        t = Transition(source="s", action="a", target="t",
                       probability=-0.1, cost=0.0)
        assert len(t.validate()) > 0

    def test_negative_cost(self):
        """Negative cost should produce a validation error."""
        t = Transition(source="s", action="a", target="t",
                       probability=0.5, cost=-1.0)
        assert any("cost" in e for e in t.validate())

    def test_zero_cost_valid(self):
        """Zero cost is a valid (free) transition."""
        t = Transition(source="s", action="a", target="t",
                       probability=1.0, cost=0.0)
        assert t.validate() == []

    def test_empty_source(self):
        """Empty source string should be flagged."""
        t = Transition(source="", action="a", target="t", probability=1.0)
        assert any("source" in e for e in t.validate())


# ═══════════════════════════════════════════════════════════════════════════
# MDP tests
# ═══════════════════════════════════════════════════════════════════════════


class TestMDP:
    """Tests for the MDP class — the full Markov Decision Process with
    transition indexing, graph queries, validation, and statistics."""

    def test_post_init_builds_index(self):
        """__post_init__ should build transition_matrix and _predecessors."""
        mdp = make_two_state_mdp()
        assert "start" in mdp.transition_matrix
        assert "go" in mdp.transition_matrix["start"]

    def test_n_states(self):
        """n_states property should match len(states)."""
        mdp = make_two_state_mdp()
        assert mdp.n_states == 2

    def test_n_actions(self):
        """n_actions property should match len(actions)."""
        mdp = make_two_state_mdp()
        assert mdp.n_actions == 1

    def test_n_transitions(self):
        """n_transitions property should match len(transitions)."""
        mdp = make_cyclic_mdp()
        assert mdp.n_transitions == 5

    def test_add_transition(self):
        """add_transition appends to list and updates indices."""
        mdp = make_two_state_mdp()
        old_count = mdp.n_transitions
        t = Transition(source="goal", action="go", target="start",
                       probability=1.0, cost=0.5)
        mdp.add_transition(t)
        assert mdp.n_transitions == old_count + 1
        assert ("start", 1.0, 0.5) in mdp.get_transitions("goal", "go")

    def test_get_actions(self):
        """get_actions returns action IDs available from a state."""
        mdp = make_cyclic_mdp()
        actions = mdp.get_actions("s2")
        assert "next" in actions
        assert "finish" in actions

    def test_get_actions_terminal(self):
        """Terminal/goal states typically have no outgoing actions."""
        mdp = make_two_state_mdp()
        assert mdp.get_actions("goal") == []

    def test_get_transitions(self):
        """get_transitions returns (target, prob, cost) triples."""
        mdp = make_two_state_mdp()
        outcomes = mdp.get_transitions("start", "go")
        assert len(outcomes) == 1
        target, prob, cost = outcomes[0]
        assert target == "goal"
        assert prob == 1.0
        assert cost == 1.0

    def test_get_successors(self):
        """get_successors returns all states reachable in one step."""
        mdp = make_cyclic_mdp()
        succs = mdp.get_successors("s2")
        assert "s0" in succs
        assert "goal" in succs

    def test_get_predecessors(self):
        """get_predecessors returns states that can reach the target in one step."""
        mdp = make_cyclic_mdp()
        preds = mdp.get_predecessors("goal")
        assert "s2" in preds

    def test_is_reachable_initial(self):
        """The initial state is always reachable from itself."""
        mdp = make_two_state_mdp()
        assert mdp.is_reachable("start") is True

    def test_is_reachable_goal(self):
        """Goal should be reachable from initial state in a connected MDP."""
        mdp = make_two_state_mdp()
        assert mdp.is_reachable("goal") is True

    def test_is_reachable_disconnected(self):
        """A state with no incoming edges from the initial state is unreachable."""
        mdp = make_two_state_mdp()
        mdp.states["island"] = State(state_id="island")
        assert mdp.is_reachable("island") is False

    def test_reachable_states(self):
        """reachable_states returns the BFS frontier from initial_state."""
        mdp = make_cyclic_mdp()
        reachable = mdp.reachable_states()
        assert "s0" in reachable
        assert "goal" in reachable
        assert len(reachable) == 4  # s0, s1, s2, goal

    def test_validate_clean_mdp(self):
        """A well-constructed MDP has no validation errors."""
        mdp = make_two_state_mdp()
        assert mdp.validate() == []

    def test_validate_bad_initial_state(self):
        """Missing initial state should be flagged."""
        mdp = make_two_state_mdp()
        mdp.initial_state = "nonexistent"
        errors = mdp.validate()
        assert any("initial_state" in e for e in errors)

    def test_validate_bad_goal_state(self):
        """A goal state not in the states dict should be flagged."""
        mdp = make_two_state_mdp()
        mdp.goal_states.add("phantom")
        errors = mdp.validate()
        assert any("phantom" in e for e in errors)

    def test_validate_probability_normalisation(self):
        """Transition probabilities that don't sum to 1 should be flagged."""
        states = {
            "a": State(state_id="a"),
            "b": State(state_id="b"),
        }
        actions = {"go": Action(action_id="go", action_type="tab")}
        transitions = [
            Transition(source="a", action="go", target="b",
                       probability=0.5, cost=0.0),
        ]
        mdp = MDP(states=states, actions=actions, transitions=transitions,
                   initial_state="a", goal_states={"b"})
        errors = mdp.validate()
        assert any("probabilities sum" in e for e in errors)

    def test_statistics_basic(self):
        """statistics() returns an MDPStatistics with correct counts."""
        mdp = make_two_state_mdp()
        stats = mdp.statistics()
        assert isinstance(stats, MDPStatistics)
        assert stats.n_states == 2
        assert stats.n_actions == 1
        assert stats.n_transitions == 1

    def test_statistics_ergodic(self):
        """A fully connected MDP where all states reach the goal is ergodic."""
        mdp = make_two_state_mdp()
        stats = mdp.statistics()
        assert stats.is_ergodic is True

    def test_statistics_branching_factor(self):
        """Branching factor is the average number of successors per state."""
        mdp = make_choice_mdp(n_choices=5)
        stats = mdp.statistics()
        # start → goal (1 successor), goal → none (0 successors)
        assert stats.branching_factor == pytest.approx(0.5, abs=0.01)

    def test_statistics_diameter(self):
        """Diameter is the longest shortest-path from the initial state."""
        mdp = make_large_chain_mdp(n=10)
        stats = mdp.statistics()
        assert stats.diameter == 9

    def test_choice_mdp_structure(self):
        """make_choice_mdp creates n_choices actions from start to goal."""
        mdp = make_choice_mdp(n_choices=3)
        assert mdp.n_states == 2
        assert mdp.n_actions == 3
        assert mdp.n_transitions == 3

    def test_cyclic_mdp_validate(self):
        """The cyclic MDP fixture should validate cleanly."""
        mdp = make_cyclic_mdp()
        assert mdp.validate() == []

    def test_discount_default(self):
        """Default discount factor should be 0.99."""
        mdp = make_two_state_mdp()
        assert mdp.discount == 0.99
