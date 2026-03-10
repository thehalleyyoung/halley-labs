"""
usability_oracle.mdp.factored — Factored MDP/POMDP representations.

Represents large UI state spaces as products of smaller *factors*,
exploiting conditional independence for compact storage and efficient
computation.

A factored MDP decomposes the state space S = S₁ × S₂ × … × S_k using
a dynamic Bayesian network (DBN) to encode structured transitions:

    T(s'_i | s_pa(i), a)

where pa(i) ⊂ {1, …, k} are the *parents* of factor i in the DBN.

Example factors for a UI:
  - scroll_position ∈ {top, middle, bottom}
  - focus_element ∈ {btn1, btn2, input1, …}
  - expanded_sections ∈ powerset of collapsible sections
  - modal_state ∈ {none, dialog1, dialog2, …}

References
----------
- Boutilier, C., Dearden, R. & Goldszmidt, M. (2000). Stochastic
  dynamic programming with factored representations. *AIJ*.
- Guestrin, C. et al. (2003). Efficient solution algorithms for factored
  MDPs. *JAIR*.
- Hoey, J. et al. (1999). Decision-theoretic planning with factored
  representations. *UAI*.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

import numpy as np

from usability_oracle.mdp.models import Action, MDP, State, Transition
from usability_oracle.mdp.pomdp import BeliefState, POMDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Factor definition
# ---------------------------------------------------------------------------


@dataclass
class Factor:
    """A single factor (state variable) in a factored MDP.

    Parameters
    ----------
    name : str
        Factor name (e.g. ``"scroll_position"``).
    values : list[str]
        Possible values (e.g. ``["top", "middle", "bottom"]``).
    parents : list[str]
        Names of factors this factor depends on in the DBN.
    """

    name: str
    values: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)

    @property
    def n_values(self) -> int:
        return len(self.values)

    def __repr__(self) -> str:
        return (
            f"Factor({self.name!r}, |vals|={self.n_values}, "
            f"parents={self.parents})"
        )


# ---------------------------------------------------------------------------
# Conditional probability table (CPT)
# ---------------------------------------------------------------------------


@dataclass
class ConditionalProbTable:
    """Conditional probability table for a factor transition.

    Stores P(factor' = v' | parent_values, action) as a nested dict.

    Structure::

        table[action_id][(parent_val_1, …, parent_val_k, current_val)] -> {next_val: prob}

    Parameters
    ----------
    factor_name : str
    table : dict
        ``action_id -> condition_tuple -> {next_val: probability}``
    """

    factor_name: str
    table: dict[str, dict[tuple[str, ...], dict[str, float]]] = field(
        default_factory=lambda: defaultdict(dict)
    )

    def prob(
        self,
        action_id: str,
        parent_values: tuple[str, ...],
        current_value: str,
        next_value: str,
    ) -> float:
        """Look up transition probability P(next | parents, current, action)."""
        condition = parent_values + (current_value,)
        dist = self.table.get(action_id, {}).get(condition, {})
        return dist.get(next_value, 0.0)

    def set_prob(
        self,
        action_id: str,
        parent_values: tuple[str, ...],
        current_value: str,
        next_value: str,
        probability: float,
    ) -> None:
        """Set a transition probability entry."""
        condition = parent_values + (current_value,)
        if action_id not in self.table:
            self.table[action_id] = {}
        if condition not in self.table[action_id]:
            self.table[action_id][condition] = {}
        self.table[action_id][condition][next_value] = probability

    def set_deterministic(
        self,
        action_id: str,
        parent_values: tuple[str, ...],
        current_value: str,
        next_value: str,
    ) -> None:
        """Set a deterministic transition (probability 1)."""
        self.set_prob(action_id, parent_values, current_value, next_value, 1.0)

    def set_identity(
        self, action_id: str, factor_values: list[str]
    ) -> None:
        """Set identity transition (factor unchanged) for an action."""
        for val in factor_values:
            self.set_deterministic(action_id, (), val, val)

    def validate(self) -> list[str]:
        """Validate that distributions sum to 1 per condition."""
        errors: list[str] = []
        for aid, conditions in self.table.items():
            for cond, dist in conditions.items():
                total = sum(dist.values())
                if abs(total - 1.0) > 1e-6:
                    errors.append(
                        f"CPT({self.factor_name}, {aid}, {cond}): "
                        f"probs sum to {total:.6f}"
                    )
        return errors


# ---------------------------------------------------------------------------
# Context-specific independence
# ---------------------------------------------------------------------------


@dataclass
class ContextSpecificIndependence:
    """Represents context-specific independence (CSI) in a factored MDP.

    CSI allows certain factor transitions to be independent of specific
    parent values in certain contexts, enabling more compact representation.

    Parameters
    ----------
    factor_name : str
    contexts : list[tuple[dict[str, str], list[str]]]
        Each entry is (context_condition, independent_parents):
        - context_condition: dict of factor_name → required value
        - independent_parents: list of parent factors that are irrelevant
          in this context.
    """

    factor_name: str
    contexts: list[tuple[dict[str, str], list[str]]] = field(default_factory=list)

    def is_independent(
        self, parent_name: str, context: dict[str, str]
    ) -> bool:
        """Check if factor is independent of parent in a given context."""
        for ctx_cond, indep_parents in self.contexts:
            if parent_name not in indep_parents:
                continue
            # Check if context matches
            match = True
            for k, v in ctx_cond.items():
                if context.get(k) != v:
                    match = False
                    break
            if match:
                return True
        return False


# ---------------------------------------------------------------------------
# Factored MDP
# ---------------------------------------------------------------------------


@dataclass
class FactoredMDP:
    """A factored MDP with structured state space and transitions.

    The state space is the Cartesian product of factor domains:
        S = S₁ × S₂ × … × S_k

    Transitions are encoded per-factor via conditional probability
    tables conditioned on parent factors (DBN structure).

    Parameters
    ----------
    factors : list[Factor]
    actions : dict[str, Action]
    cpts : dict[str, ConditionalProbTable]
        Factor name → CPT.
    reward_factors : dict[str, dict[tuple[str, ...], float]]
        Factor name → {(value_tuple) → reward}.
    discount : float
    """

    factors: list[Factor] = field(default_factory=list)
    actions: dict[str, Action] = field(default_factory=dict)
    cpts: dict[str, ConditionalProbTable] = field(default_factory=dict)
    reward_factors: dict[str, dict[tuple[str, ...], float]] = field(
        default_factory=dict
    )
    discount: float = 0.99
    csi: dict[str, ContextSpecificIndependence] = field(default_factory=dict)

    @property
    def n_factors(self) -> int:
        return len(self.factors)

    @property
    def n_flat_states(self) -> int:
        """Total number of states in the flat (expanded) representation."""
        if not self.factors:
            return 0
        n = 1
        for f in self.factors:
            n *= f.n_values
        return n

    @property
    def factor_names(self) -> list[str]:
        return [f.name for f in self.factors]

    def get_factor(self, name: str) -> Optional[Factor]:
        """Look up a factor by name."""
        for f in self.factors:
            if f.name == name:
                return f
        return None

    def factor_state(
        self, flat_state: dict[str, str]
    ) -> tuple[str, ...]:
        """Convert a dict-based state to an ordered tuple."""
        return tuple(flat_state.get(f.name, f.values[0]) for f in self.factors)

    def state_dict(
        self, factor_tuple: tuple[str, ...]
    ) -> dict[str, str]:
        """Convert an ordered tuple to a dict-based state."""
        return {f.name: factor_tuple[i] for i, f in enumerate(self.factors)}

    def transition_prob(
        self,
        state: dict[str, str],
        action_id: str,
        next_state: dict[str, str],
    ) -> float:
        """Compute joint transition probability by factoring.

        T(s'|s,a) = Π_i T_i(s'_i | s_{pa(i)}, s_i, a)

        Parameters
        ----------
        state : dict[str, str]
        action_id : str
        next_state : dict[str, str]

        Returns
        -------
        float
        """
        prob = 1.0
        for factor in self.factors:
            cpt = self.cpts.get(factor.name)
            if cpt is None:
                # Identity transition
                if state.get(factor.name) != next_state.get(factor.name):
                    return 0.0
                continue

            parent_vals = tuple(state.get(p, "") for p in factor.parents)
            current_val = state.get(factor.name, factor.values[0])
            next_val = next_state.get(factor.name, factor.values[0])

            p = cpt.prob(action_id, parent_vals, current_val, next_val)
            prob *= p

            if prob <= 0:
                return 0.0

        return prob

    def factored_reward(self, state: dict[str, str], action_id: str) -> float:
        """Compute reward as sum of factor rewards.

        R(s, a) = Σ_f R_f(s_f, s_{pa(f)})

        Parameters
        ----------
        state : dict[str, str]
        action_id : str

        Returns
        -------
        float
        """
        total = 0.0
        for fname, reward_table in self.reward_factors.items():
            factor = self.get_factor(fname)
            if factor is None:
                continue
            key_vals = tuple(state.get(p, "") for p in factor.parents)
            key_vals += (state.get(fname, ""),)
            total += reward_table.get(key_vals, 0.0)
        return total

    def to_flat_mdp(self) -> MDP:
        """Expand the factored MDP into a flat (tabular) MDP.

        Warning: exponential in the number of factors.

        Returns
        -------
        MDP
        """
        factor_values = [f.values for f in self.factors]
        all_states = list(itertools.product(*factor_values))

        states: dict[str, State] = {}
        for state_tuple in all_states:
            sid = ":".join(state_tuple)
            state_dict = self.state_dict(state_tuple)
            features = {f.name: float(f.values.index(state_tuple[i]))
                        for i, f in enumerate(self.factors)}
            states[sid] = State(
                state_id=sid,
                features=features,
                label=sid,
                metadata=state_dict,
            )

        transitions: list[Transition] = []
        for state_tuple in all_states:
            src_id = ":".join(state_tuple)
            src_dict = self.state_dict(state_tuple)

            for aid in self.actions:
                for next_tuple in all_states:
                    next_id = ":".join(next_tuple)
                    next_dict = self.state_dict(next_tuple)

                    prob = self.transition_prob(src_dict, aid, next_dict)
                    if prob > 1e-15:
                        cost = -self.factored_reward(src_dict, aid)
                        transitions.append(Transition(
                            source=src_id,
                            action=aid,
                            target=next_id,
                            probability=prob,
                            cost=max(0.0, cost),
                        ))

        initial = ":".join(self.factors[0].values[0] if f.values else "" for f in self.factors)

        return MDP(
            states=states,
            actions=dict(self.actions),
            transitions=transitions,
            initial_state=initial,
            discount=self.discount,
        )

    def validate(self) -> list[str]:
        """Validate the factored MDP."""
        errors: list[str] = []

        # Check factor parents exist
        fnames = {f.name for f in self.factors}
        for factor in self.factors:
            for p in factor.parents:
                if p not in fnames:
                    errors.append(
                        f"Factor {factor.name!r}: parent {p!r} not found"
                    )

        # Check CPTs
        for fname, cpt in self.cpts.items():
            if fname not in fnames:
                errors.append(f"CPT for unknown factor {fname!r}")
            errors.extend(cpt.validate())

        return errors

    def __repr__(self) -> str:
        return (
            f"FactoredMDP(factors={self.n_factors}, "
            f"|S|={self.n_flat_states}, γ={self.discount})"
        )


# ---------------------------------------------------------------------------
# Factored value function
# ---------------------------------------------------------------------------


@dataclass
class FactoredValueFunction:
    """Value function decomposed over factor subsets.

    V(s) ≈ Σ_C V_C(s_C)

    where each V_C depends only on a subset C of factors.

    Parameters
    ----------
    components : dict[tuple[str, ...], dict[tuple[str, ...], float]]
        Mapping factor_subset → {value_assignment → V_C}.
    """

    components: dict[tuple[str, ...], dict[tuple[str, ...], float]] = field(
        default_factory=dict
    )

    def value(self, state: dict[str, str]) -> float:
        """Evaluate V(s) = Σ_C V_C(s_C)."""
        total = 0.0
        for factor_subset, value_table in self.components.items():
            key = tuple(state.get(f, "") for f in factor_subset)
            total += value_table.get(key, 0.0)
        return total

    def update_component(
        self,
        factor_subset: tuple[str, ...],
        assignment: tuple[str, ...],
        value: float,
    ) -> None:
        """Update a single component value."""
        if factor_subset not in self.components:
            self.components[factor_subset] = {}
        self.components[factor_subset][assignment] = value

    @property
    def n_components(self) -> int:
        return len(self.components)

    @property
    def total_entries(self) -> int:
        return sum(len(vt) for vt in self.components.values())

    def __repr__(self) -> str:
        return (
            f"FactoredValueFunction(components={self.n_components}, "
            f"entries={self.total_entries})"
        )


# ---------------------------------------------------------------------------
# Algebraic decision diagram (simplified)
# ---------------------------------------------------------------------------


@dataclass
class ADDNode:
    """A node in an algebraic decision diagram (ADD).

    ADDs are compact representations of functions over discrete variables.
    Internal nodes test a variable; leaves hold a value.

    Parameters
    ----------
    variable : str or None
        Variable tested at this node (None for leaves).
    children : dict[str, ADDNode]
        Value → child node.
    value : float or None
        Leaf value (None for internal nodes).
    """

    variable: Optional[str] = None
    children: dict[str, ADDNode] = field(default_factory=dict)
    value: Optional[float] = None

    @property
    def is_leaf(self) -> bool:
        return self.variable is None

    def evaluate(self, assignment: dict[str, str]) -> float:
        """Evaluate the ADD for a variable assignment."""
        if self.is_leaf:
            return self.value if self.value is not None else 0.0
        val = assignment.get(self.variable, "")  # type: ignore[arg-type]
        child = self.children.get(val)
        if child is None:
            # Default: first child or 0
            if self.children:
                child = next(iter(self.children.values()))
            else:
                return 0.0
        return child.evaluate(assignment)

    @property
    def size(self) -> int:
        """Number of nodes in this ADD."""
        if self.is_leaf:
            return 1
        return 1 + sum(c.size for c in self.children.values())

    def __repr__(self) -> str:
        if self.is_leaf:
            return f"ADDLeaf({self.value})"
        return f"ADDNode({self.variable!r}, children={len(self.children)})"


def build_add_from_table(
    variables: list[str],
    variable_values: dict[str, list[str]],
    value_table: dict[tuple[str, ...], float],
) -> ADDNode:
    """Build an ADD from a value table.

    Constructs an ADD by recursively splitting on variables in order.
    Merges subtrees that produce identical values for compactness.

    Parameters
    ----------
    variables : list[str]
        Variable ordering for the ADD.
    variable_values : dict[str, list[str]]
        Possible values for each variable.
    value_table : dict[tuple[str, ...], float]
        (value_1, …, value_k) → function value.

    Returns
    -------
    ADDNode
    """
    if not variables:
        # Leaf: should be a single value
        if value_table:
            val = next(iter(value_table.values()))
            return ADDNode(value=val)
        return ADDNode(value=0.0)

    var = variables[0]
    remaining = variables[1:]
    vals = variable_values.get(var, [])

    children: dict[str, ADDNode] = {}
    for v in vals:
        # Filter table entries matching this value
        sub_table: dict[tuple[str, ...], float] = {}
        for key, fval in value_table.items():
            if key and key[0] == v:
                sub_table[key[1:]] = fval

        child = build_add_from_table(remaining, variable_values, sub_table)
        children[v] = child

    # Check if all children are identical leaves → merge
    if all(c.is_leaf for c in children.values()):
        vals_set = {c.value for c in children.values()}
        if len(vals_set) == 1:
            return ADDNode(value=vals_set.pop())

    return ADDNode(variable=var, children=children)


# ---------------------------------------------------------------------------
# Factored belief update
# ---------------------------------------------------------------------------


def factored_belief_update(
    factored_mdp: FactoredMDP,
    belief_factors: dict[str, dict[str, float]],
    action_id: str,
) -> dict[str, dict[str, float]]:
    """Update a factored belief state given an action.

    Assumes conditional independence between factors (mean-field
    approximation).

    b'_i(v') = Σ_v T_i(v'|pa_values, v, a) · b_i(v)

    where pa_values are computed from the current factored belief
    (using the mode of each parent's marginal).

    Parameters
    ----------
    factored_mdp : FactoredMDP
    belief_factors : dict[str, dict[str, float]]
        Factor name → {value → probability}.
    action_id : str

    Returns
    -------
    dict[str, dict[str, float]]
        Updated factored belief.
    """
    # Compute mode (most likely value) for each factor
    modes: dict[str, str] = {}
    for fname, dist in belief_factors.items():
        if dist:
            modes[fname] = max(dist, key=dist.get)  # type: ignore[arg-type]
        else:
            factor = factored_mdp.get_factor(fname)
            modes[fname] = factor.values[0] if factor and factor.values else ""

    new_beliefs: dict[str, dict[str, float]] = {}

    for factor in factored_mdp.factors:
        cpt = factored_mdp.cpts.get(factor.name)
        current_dist = belief_factors.get(factor.name, {})

        if cpt is None:
            # Identity transition
            new_beliefs[factor.name] = dict(current_dist)
            continue

        parent_vals = tuple(modes.get(p, "") for p in factor.parents)

        new_dist: dict[str, float] = {}
        for next_val in factor.values:
            prob = 0.0
            for curr_val, b_curr in current_dist.items():
                if b_curr <= 0:
                    continue
                p_trans = cpt.prob(action_id, parent_vals, curr_val, next_val)
                prob += p_trans * b_curr
            if prob > 1e-15:
                new_dist[next_val] = prob

        # Normalise
        total = sum(new_dist.values())
        if total > 0:
            new_dist = {v: p / total for v, p in new_dist.items()}
        new_beliefs[factor.name] = new_dist

    return new_beliefs


# ---------------------------------------------------------------------------
# UI state space factoring
# ---------------------------------------------------------------------------


class UIStateFactorBuilder:
    """Build a factored MDP from common UI state decompositions.

    Standard UI factors:
    - ``scroll_position``: viewport scroll state
    - ``focus_element``: currently focused interactive element
    - ``expanded_collapsed``: collapsible section states
    - ``modal_state``: active modal/dialog
    - ``form_state``: form field completion status

    Parameters
    ----------
    tree : AccessibilityTree (duck-typed)
    """

    def __init__(self, tree: Any = None) -> None:
        self.tree = tree

    def build_factors(
        self,
        scroll_positions: Optional[list[str]] = None,
        focus_elements: Optional[list[str]] = None,
        collapsible_sections: Optional[list[str]] = None,
        modal_names: Optional[list[str]] = None,
    ) -> list[Factor]:
        """Build factor list from UI structure.

        Parameters
        ----------
        scroll_positions : list[str], optional
        focus_elements : list[str], optional
        collapsible_sections : list[str], optional
        modal_names : list[str], optional

        Returns
        -------
        list[Factor]
        """
        factors: list[Factor] = []

        if scroll_positions:
            factors.append(Factor(
                name="scroll_position",
                values=scroll_positions,
                parents=[],
            ))

        if focus_elements:
            factors.append(Factor(
                name="focus_element",
                values=focus_elements,
                parents=[],
            ))

        if collapsible_sections:
            # Each section can be expanded or collapsed
            for section in collapsible_sections:
                factors.append(Factor(
                    name=f"section_{section}",
                    values=["collapsed", "expanded"],
                    parents=[],
                ))

        if modal_names:
            factors.append(Factor(
                name="modal_state",
                values=["none"] + modal_names,
                parents=[],
            ))

        return factors

    def build_factored_mdp(
        self,
        factors: list[Factor],
        actions: dict[str, Action],
    ) -> FactoredMDP:
        """Build a factored MDP with default (identity) transitions.

        Specific transitions should be added via CPTs after construction.

        Parameters
        ----------
        factors : list[Factor]
        actions : dict[str, Action]

        Returns
        -------
        FactoredMDP
        """
        cpts: dict[str, ConditionalProbTable] = {}

        for factor in factors:
            cpt = ConditionalProbTable(factor_name=factor.name)
            # Default: identity transition for all actions
            for aid in actions:
                cpt.set_identity(aid, factor.values)
            cpts[factor.name] = cpt

        return FactoredMDP(
            factors=factors,
            actions=actions,
            cpts=cpts,
        )
