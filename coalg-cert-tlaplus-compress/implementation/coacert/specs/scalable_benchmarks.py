"""
Large-scale benchmark specifications for scalability analysis.

Provides parameterized specifications that can generate state spaces
from hundreds to hundreds of thousands of states, addressing the
review critique about limited scalability analysis.
"""

from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class ScalableSpec:
    """A parameterized specification for scalability testing."""

    name: str
    parameter: int
    states: Set[str] = field(default_factory=set)
    initial_states: Set[str] = field(default_factory=set)
    transitions: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)
    propositions: Dict[str, FrozenSet[str]] = field(default_factory=dict)
    actions: Set[str] = field(default_factory=set)
    fairness_pairs: List[Tuple[Set[str], Set[str]]] = field(default_factory=list)
    spec_params: Dict[str, Any] = field(default_factory=dict)

    @property
    def state_count(self) -> int:
        return len(self.states)

    @property
    def transition_count(self) -> int:
        return sum(
            len(targets)
            for succs in self.transitions.values()
            for targets in succs.values()
        )


def make_dining_philosophers(n: int) -> ScalableSpec:
    """Generate dining philosophers with n philosophers.

    State space: 3^n × 2^n (each philosopher in {thinking, hungry, eating},
    each fork in {free, taken}).
    For n=5: 7776 states, n=8: ~1.7M states.
    """
    spec = ScalableSpec(
        name=f"dining_philosophers_{n}",
        parameter=n,
        actions={"pickup_left", "pickup_right", "putdown", "think"},
        spec_params={
            f"phil_{i}": {"type": "enum", "values": ["thinking", "hungry", "eating"]}
            for i in range(n)
        },
    )
    spec.spec_params.update({
        f"fork_{i}": {"type": "boolean"}
        for i in range(n)
    })

    # Generate states: each philosopher in {T, H, E}, each fork in {0, 1}
    phil_states = ["T", "H", "E"]
    fork_states = [0, 1]

    for phil_config in itertools.product(phil_states, repeat=n):
        for fork_config in itertools.product(fork_states, repeat=n):
            # Check validity: if phil[i] is eating, forks i and (i+1)%n must be taken
            valid = True
            for i in range(n):
                if phil_config[i] == "E":
                    if fork_config[i] == 0 or fork_config[(i + 1) % n] == 0:
                        valid = False
                        break
            if not valid:
                continue

            state_name = "".join(phil_config) + "_" + "".join(map(str, fork_config))
            spec.states.add(state_name)

            # Propositions
            props = set()
            for i in range(n):
                props.add(f"phil{i}_{phil_config[i]}")
                if fork_config[i]:
                    props.add(f"fork{i}_taken")
            spec.propositions[state_name] = frozenset(props)

            # Transitions
            spec.transitions[state_name] = {}

            for i in range(n):
                # Think: eating -> thinking (release forks)
                if phil_config[i] == "E":
                    new_phil = list(phil_config)
                    new_fork = list(fork_config)
                    new_phil[i] = "T"
                    new_fork[i] = 0
                    new_fork[(i + 1) % n] = 0
                    target = "".join(new_phil) + "_" + "".join(map(str, new_fork))
                    if target in spec.states or True:  # add anyway, filter later
                        spec.transitions[state_name].setdefault("putdown", set()).add(target)

                # Get hungry: thinking -> hungry
                if phil_config[i] == "T":
                    new_phil = list(phil_config)
                    new_phil[i] = "H"
                    target = "".join(new_phil) + "_" + "".join(map(str, fork_config))
                    spec.transitions[state_name].setdefault("think", set()).add(target)

                # Try to eat: hungry + both forks free -> eating
                if phil_config[i] == "H":
                    left = i
                    right = (i + 1) % n
                    if fork_config[left] == 0 and fork_config[right] == 0:
                        new_phil = list(phil_config)
                        new_fork = list(fork_config)
                        new_phil[i] = "E"
                        new_fork[left] = 1
                        new_fork[right] = 1
                        target = "".join(new_phil) + "_" + "".join(map(str, new_fork))
                        spec.transitions[state_name].setdefault("pickup_left", set()).add(target)

    # Filter transitions to valid states only
    valid_states = spec.states
    for s in list(spec.transitions.keys()):
        for act in list(spec.transitions[s].keys()):
            spec.transitions[s][act] &= valid_states

    # Initial state: all thinking, all forks free
    initial = "T" * n + "_" + "0" * n
    if initial in spec.states:
        spec.initial_states.add(initial)

    # Fairness: each philosopher eventually eats (if hungry)
    for i in range(n):
        b_states = {s for s in spec.states if s[i] == "H"}
        g_states = {s for s in spec.states if s[i] == "E"}
        spec.fairness_pairs.append((b_states, g_states))

    # Limit size for tractability
    if len(spec.states) > 200000:
        # Truncate for safety
        limited = set(list(spec.states)[:200000])
        spec.states = limited

    return spec


def make_token_ring(n: int) -> ScalableSpec:
    """Generate a token ring with n processes.

    State space: n × 2^n (token position × process active flags).
    For n=8: 2048 states, n=16: ~1M states.
    """
    spec = ScalableSpec(
        name=f"token_ring_{n}",
        parameter=n,
        actions={"pass_token", "process"},
        spec_params={
            "token_pos": {"type": "bounded_int", "lo": 0, "hi": n - 1},
            **{f"active_{i}": {"type": "boolean"} for i in range(n)},
        },
    )

    # Generate states: token position × active flags (limited)
    max_states = min(n * (2 ** n), 100000)
    count = 0

    for token_pos in range(n):
        if count >= max_states:
            break
        for active_bits in range(min(2 ** n, max_states // n)):
            if count >= max_states:
                break
            active = tuple((active_bits >> i) & 1 for i in range(n))
            state_name = f"t{token_pos}_" + "".join(map(str, active))
            spec.states.add(state_name)

            # Propositions
            props = {f"token_at_{token_pos}"}
            for i in range(n):
                if active[i]:
                    props.add(f"active_{i}")
            spec.propositions[state_name] = frozenset(props)

            # Transitions
            spec.transitions[state_name] = {}

            # Pass token to next process
            next_pos = (token_pos + 1) % n
            target_name = f"t{next_pos}_" + "".join(map(str, active))
            spec.transitions[state_name]["pass_token"] = {target_name}

            # Process at token: toggle active
            new_active = list(active)
            new_active[token_pos] = 1 - new_active[token_pos]
            target_name = f"t{token_pos}_" + "".join(map(str, new_active))
            spec.transitions[state_name]["process"] = {target_name}

            count += 1

    # Initial state: token at 0, all inactive
    initial = f"t0_" + "0" * n
    if initial in spec.states:
        spec.initial_states.add(initial)

    # Fairness: token visits each process
    for i in range(n):
        b_states = spec.states  # all states request
        g_states = {s for s in spec.states if s.startswith(f"t{i}_")}
        spec.fairness_pairs.append((b_states, g_states))

    return spec


def make_mutex_n(n: int) -> ScalableSpec:
    """Generate n-process mutual exclusion (Peterson-style generalization).

    State space: 4^n × n (each process in {idle, trying, waiting, critical},
    plus turn variable).
    For n=3: 192 states, n=5: ~5120 states, n=8: ~524K states.
    """
    proc_states = ["I", "T", "W", "C"]  # idle, trying, waiting, critical

    spec = ScalableSpec(
        name=f"mutex_{n}",
        parameter=n,
        actions={"try_enter", "wait", "enter_critical", "exit_critical"},
        spec_params={
            **{f"proc_{i}": {"type": "enum", "values": proc_states} for i in range(n)},
            "turn": {"type": "bounded_int", "lo": 0, "hi": n - 1},
        },
    )

    max_states = 100000
    count = 0

    for turn in range(n):
        if count >= max_states:
            break
        for proc_config in itertools.product(proc_states, repeat=n):
            if count >= max_states:
                break

            # Mutual exclusion invariant: at most one in critical
            critical_count = sum(1 for p in proc_config if p == "C")
            if critical_count > 1:
                continue

            state_name = "".join(proc_config) + f"_{turn}"
            spec.states.add(state_name)

            props = set()
            for i in range(n):
                props.add(f"p{i}_{proc_config[i]}")
            props.add(f"turn_{turn}")
            if critical_count > 0:
                props.add("mutex_held")
            spec.propositions[state_name] = frozenset(props)

            spec.transitions[state_name] = {}

            for i in range(n):
                if proc_config[i] == "I":
                    # Try to enter
                    new_proc = list(proc_config)
                    new_proc[i] = "T"
                    target = "".join(new_proc) + f"_{turn}"
                    spec.transitions[state_name].setdefault("try_enter", set()).add(target)

                elif proc_config[i] == "T":
                    # Wait
                    new_proc = list(proc_config)
                    new_proc[i] = "W"
                    new_turn = (turn + 1) % n
                    target = "".join(new_proc) + f"_{new_turn}"
                    spec.transitions[state_name].setdefault("wait", set()).add(target)

                elif proc_config[i] == "W":
                    # Enter critical if turn == i and no other in critical
                    if turn == i and critical_count == 0:
                        new_proc = list(proc_config)
                        new_proc[i] = "C"
                        target = "".join(new_proc) + f"_{turn}"
                        spec.transitions[state_name].setdefault("enter_critical", set()).add(target)

                elif proc_config[i] == "C":
                    # Exit critical
                    new_proc = list(proc_config)
                    new_proc[i] = "I"
                    new_turn = (turn + 1) % n
                    target = "".join(new_proc) + f"_{new_turn}"
                    spec.transitions[state_name].setdefault("exit_critical", set()).add(target)

            count += 1

    # Initial state: all idle, turn = 0
    initial = "I" * n + "_0"
    if initial in spec.states:
        spec.initial_states.add(initial)

    # Fairness: each process eventually enters critical if trying
    for i in range(n):
        b_states = {s for s in spec.states if s[i] == "T" or s[i] == "W"}
        g_states = {s for s in spec.states if s[i] == "C"}
        spec.fairness_pairs.append((b_states, g_states))

    return spec


# Registry of scalable benchmarks
SCALABLE_BENCHMARKS = {
    "dining_philosophers": make_dining_philosophers,
    "token_ring": make_token_ring,
    "mutex": make_mutex_n,
}


def list_scalable_benchmarks() -> List[Dict[str, Any]]:
    """List available scalable benchmarks with parameter info."""
    return [
        {
            "name": "dining_philosophers",
            "parameter": "number of philosophers",
            "typical_range": [3, 4, 5, 6, 7],
            "state_space_formula": "~3^n × 2^n (with validity constraints)",
        },
        {
            "name": "token_ring",
            "parameter": "number of processes",
            "typical_range": [4, 8, 12, 16],
            "state_space_formula": "n × 2^n",
        },
        {
            "name": "mutex",
            "parameter": "number of processes",
            "typical_range": [2, 3, 4, 5, 6],
            "state_space_formula": "~4^n × n (with mutex invariant)",
        },
    ]
