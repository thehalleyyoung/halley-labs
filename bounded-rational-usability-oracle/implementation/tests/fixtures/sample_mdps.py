"""Sample MDPs for testing."""

from __future__ import annotations

from usability_oracle.mdp.models import State, Action, Transition, MDP


def make_two_state_mdp() -> MDP:
    """Minimal 2-state MDP: start -> goal."""
    states = {
        "start": State(state_id="start", features={"x": 0.0}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={"x": 1.0}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {
        "go": Action(action_id="go", action_type="click",
                     target_node_id="g", description="go", preconditions=[]),
    }
    transitions = [
        Transition(source="start", action="go", target="goal",
                   probability=1.0, cost=1.0),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)


def make_cyclic_mdp() -> MDP:
    """A 3-state MDP with a cycle: s0 -> s1 -> s2 -> s0 and s2 -> goal."""
    states = {
        f"s{i}": State(state_id=f"s{i}", features={"pos": float(i)},
                       label=f"s{i}", is_terminal=False, is_goal=False,
                       metadata={})
        for i in range(3)
    }
    states["goal"] = State(state_id="goal", features={"pos": 3.0},
                           label="goal", is_terminal=True, is_goal=True,
                           metadata={})
    actions = {
        "next": Action(action_id="next", action_type="click",
                       target_node_id="n", description="next", preconditions=[]),
        "finish": Action(action_id="finish", action_type="click",
                         target_node_id="f", description="finish", preconditions=[]),
    }
    transitions = [
        Transition(source="s0", action="next", target="s1", probability=1.0, cost=0.3),
        Transition(source="s1", action="next", target="s2", probability=1.0, cost=0.3),
        Transition(source="s2", action="next", target="s0", probability=0.5, cost=0.3),
        Transition(source="s2", action="finish", target="goal", probability=1.0, cost=0.1),
        Transition(source="s2", action="next", target="goal", probability=0.5, cost=0.3),
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={"goal"}, discount=0.99)


def make_large_chain_mdp(n: int = 20) -> MDP:
    """A linear chain MDP with n states."""
    states = {}
    for i in range(n):
        states[f"s{i}"] = State(
            state_id=f"s{i}", features={"pos": float(i)}, label=f"s{i}",
            is_terminal=(i == n - 1), is_goal=(i == n - 1), metadata={},
        )
    actions = {
        "step": Action(action_id="step", action_type="click",
                       target_node_id="x", description="step", preconditions=[]),
    }
    transitions = [
        Transition(source=f"s{i}", action="step", target=f"s{i+1}",
                   probability=1.0, cost=0.5)
        for i in range(n - 1)
    ]
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="s0", goal_states={f"s{n-1}"}, discount=0.99)


def make_choice_mdp(n_choices: int = 5) -> MDP:
    """An MDP with a single state and n_choices actions leading to goal."""
    states = {
        "start": State(state_id="start", features={}, label="start",
                       is_terminal=False, is_goal=False, metadata={}),
        "goal": State(state_id="goal", features={}, label="goal",
                      is_terminal=True, is_goal=True, metadata={}),
    }
    actions = {}
    transitions = []
    for i in range(n_choices):
        aid = f"choice_{i}"
        actions[aid] = Action(action_id=aid, action_type="click",
                              target_node_id=f"c{i}", description=f"Choice {i}",
                              preconditions=[])
        transitions.append(
            Transition(source="start", action=aid, target="goal",
                       probability=1.0, cost=0.3 + 0.1 * i)
        )
    return MDP(states=states, actions=actions, transitions=transitions,
               initial_state="start", goal_states={"goal"}, discount=0.99)
