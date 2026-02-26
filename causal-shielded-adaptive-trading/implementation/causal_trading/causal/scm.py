"""
Structural Causal Model (SCM) representation and inference.

Implements DAG-based causal models with support for linear and additive-noise
structural equations, do-calculus interventions, counterfactual computation,
d-separation testing, Markov blanket extraction, and Monte-Carlo simulation.
"""

from __future__ import annotations

import copy
import json
import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Sequence,
    Tuple,
    Union,
)

import networkx as nx
import numpy as np
from numpy.typing import NDArray
from scipy import stats


# ---------------------------------------------------------------------------
# Structural equation abstractions
# ---------------------------------------------------------------------------

class StructuralEquation(ABC):
    """Abstract base for a structural equation  X_j = f_j(Pa_j) + N_j."""

    @abstractmethod
    def evaluate(
        self,
        parent_values: Dict[str, float],
        noise: Optional[float] = None,
    ) -> float:
        """Compute the value of the variable given parent values and noise."""

    @abstractmethod
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the equation to a JSON-compatible dictionary."""

    @classmethod
    @abstractmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructuralEquation":
        """Deserialise from a dictionary."""


class LinearEquation(StructuralEquation):
    """Linear structural equation: X_j = sum_i (w_i * Pa_i) + intercept + N_j."""

    def __init__(
        self,
        weights: Dict[str, float],
        intercept: float = 0.0,
        noise_std: float = 1.0,
    ) -> None:
        self.weights = dict(weights)
        self.intercept = intercept
        self.noise_std = noise_std

    def evaluate(
        self,
        parent_values: Dict[str, float],
        noise: Optional[float] = None,
    ) -> float:
        val = self.intercept
        for parent, w in self.weights.items():
            val += w * parent_values.get(parent, 0.0)
        if noise is None:
            noise = np.random.normal(0, self.noise_std)
        return float(val + noise)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "linear",
            "weights": self.weights,
            "intercept": self.intercept,
            "noise_std": self.noise_std,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "LinearEquation":
        return cls(
            weights=d["weights"],
            intercept=d.get("intercept", 0.0),
            noise_std=d.get("noise_std", 1.0),
        )


class ANMEquation(StructuralEquation):
    """Additive noise model equation: X_j = f(Pa_j) + N_j.

    The function *f* is specified either as an explicit callable or as a
    combination of polynomial/RBF basis components stored for serialisation.
    """

    def __init__(
        self,
        func: Callable[..., float],
        parent_names: List[str],
        noise_std: float = 1.0,
        description: str = "custom",
        params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.func = func
        self.parent_names = list(parent_names)
        self.noise_std = noise_std
        self.description = description
        self.params = params or {}

    def evaluate(
        self,
        parent_values: Dict[str, float],
        noise: Optional[float] = None,
    ) -> float:
        args = [parent_values.get(p, 0.0) for p in self.parent_names]
        val = self.func(*args)
        if noise is None:
            noise = np.random.normal(0, self.noise_std)
        return float(val + noise)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": "anm",
            "parent_names": self.parent_names,
            "noise_std": self.noise_std,
            "description": self.description,
            "params": self.params,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ANMEquation":
        desc = d.get("description", "identity")
        parent_names = d["parent_names"]

        # Reconstruct common functional forms
        if desc == "identity":
            func = lambda *a: sum(a)
        elif desc == "quadratic":
            func = lambda *a: sum(x ** 2 for x in a)
        elif desc == "tanh":
            scale = d.get("params", {}).get("scale", 1.0)
            func = lambda *a, s=scale: s * np.tanh(sum(a))
        else:
            func = lambda *a: sum(a)

        return cls(
            func=func,
            parent_names=parent_names,
            noise_std=d.get("noise_std", 1.0),
            description=desc,
            params=d.get("params", {}),
        )


# ---------------------------------------------------------------------------
# Structural Causal Model
# ---------------------------------------------------------------------------

class StructuralCausalModel:
    """Full structural causal model over a set of named variables.

    Parameters
    ----------
    name : str
        Human-readable identifier for the model.

    Attributes
    ----------
    graph : nx.DiGraph
        Directed acyclic graph encoding causal structure.
    equations : dict[str, StructuralEquation]
        Mapping from variable name to its structural equation.
    exogenous_distributions : dict[str, stats.rv_continuous]
        Noise distributions keyed by variable name.
    """

    def __init__(self, name: str = "SCM") -> None:
        self.name = name
        self.graph: nx.DiGraph = nx.DiGraph()
        self.equations: Dict[str, StructuralEquation] = {}
        self.exogenous_distributions: Dict[str, stats.rv_continuous] = {}
        self._topological_order: Optional[List[str]] = None

    # ------------------------------------------------------------------ nodes
    def add_variable(
        self,
        name: str,
        equation: Optional[StructuralEquation] = None,
        noise_distribution: Optional[stats.rv_continuous] = None,
    ) -> None:
        """Register a new endogenous variable."""
        self.graph.add_node(name)
        if equation is not None:
            self.equations[name] = equation
        if noise_distribution is not None:
            self.exogenous_distributions[name] = noise_distribution
        self._topological_order = None

    def add_edge(self, parent: str, child: str) -> None:
        """Add a directed edge *parent* → *child*."""
        if parent not in self.graph:
            self.graph.add_node(parent)
        if child not in self.graph:
            self.graph.add_node(child)
        self.graph.add_edge(parent, child)
        if not nx.is_directed_acyclic_graph(self.graph):
            self.graph.remove_edge(parent, child)
            raise ValueError(
                f"Adding edge {parent} -> {child} would create a cycle."
            )
        self._topological_order = None

    def remove_edge(self, parent: str, child: str) -> None:
        """Remove a directed edge."""
        self.graph.remove_edge(parent, child)
        self._topological_order = None

    # ----------------------------------------------------------- properties
    @property
    def variables(self) -> List[str]:
        return list(self.graph.nodes)

    @property
    def edges(self) -> List[Tuple[str, str]]:
        return list(self.graph.edges)

    @property
    def topological_order(self) -> List[str]:
        if self._topological_order is None:
            self._topological_order = list(nx.topological_sort(self.graph))
        return self._topological_order

    def parents(self, node: str) -> List[str]:
        return list(self.graph.predecessors(node))

    def children(self, node: str) -> List[str]:
        return list(self.graph.successors(node))

    def ancestors(self, node: str) -> Set[str]:
        return nx.ancestors(self.graph, node)

    def descendants(self, node: str) -> Set[str]:
        return nx.descendants(self.graph, node)

    # -------------------------------------------------------- Markov blanket
    def markov_blanket(self, node: str) -> Set[str]:
        """Return the Markov blanket of *node*: parents, children, and
        co-parents (other parents of its children)."""
        blanket: Set[str] = set()
        blanket.update(self.parents(node))
        for child in self.children(node):
            blanket.add(child)
            blanket.update(self.parents(child))
        blanket.discard(node)
        return blanket

    # --------------------------------------------------------- d-separation
    def d_separated(
        self,
        x: Union[str, Set[str]],
        y: Union[str, Set[str]],
        z: Union[str, Set[str], None] = None,
    ) -> bool:
        """Test d-separation of *x* and *y* given *z* using the Bayes-ball
        algorithm (Shachter 1998)."""
        if isinstance(x, str):
            x = {x}
        if isinstance(y, str):
            y = {y}
        if z is None:
            z = set()
        elif isinstance(z, str):
            z = {z}

        return self._bayes_ball(x, y, z)

    def _bayes_ball(
        self, x: Set[str], y: Set[str], z: Set[str]
    ) -> bool:
        """Bayes-ball reachability: returns True iff x ⊥ y | z in the DAG."""
        # If x and y overlap, they are trivially not d-separated
        if x & y:
            return False
        reachable = self._reachable_from(x, z)
        return len(reachable & y) == 0

    def _reachable_from(
        self, sources: Set[str], conditioned: Set[str]
    ) -> Set[str]:
        """Nodes reachable from *sources* via active trails given
        *conditioned*, using the Bayes-ball algorithm."""
        ancestors_of_z = set()
        for node in conditioned:
            ancestors_of_z |= nx.ancestors(self.graph, node)
            ancestors_of_z.add(node)

        visited_up: Set[str] = set()
        visited_down: Set[str] = set()
        reachable: Set[str] = set()
        # queue entries: (node, direction) where direction is "up" or "down"
        queue: List[Tuple[str, str]] = [(s, "up") for s in sources]

        while queue:
            node, direction = queue.pop()
            if (node, direction) in {(n, d) for n, d in itertools.chain(
                ((v, "up") for v in visited_up),
                ((v, "down") for v in visited_down),
            )}:
                continue

            if direction == "up":
                visited_up.add(node)
            else:
                visited_down.add(node)

            if node not in sources:
                reachable.add(node)

            if direction == "up" and node not in conditioned:
                # Pass through – visit children (down) and parents (up)
                for parent in self.parents(node):
                    queue.append((parent, "up"))
                for child in self.children(node):
                    queue.append((child, "down"))
            elif direction == "down":
                if node not in conditioned:
                    for child in self.children(node):
                        queue.append((child, "down"))
                if node in ancestors_of_z or node in conditioned:
                    for parent in self.parents(node):
                        queue.append((parent, "up"))

        return reachable

    # ---------------------------------------------------------- simulation
    def sample(
        self,
        n: int = 1000,
        interventions: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Dict[str, NDArray[np.floating]]:
        """Forward-sample *n* observations from the SCM.

        Parameters
        ----------
        n : int
            Number of samples.
        interventions : dict, optional
            Hard interventions ``do(X=x)`` – keys are variable names, values
            are the constants to set.
        seed : int, optional
            Random seed for reproducibility.

        Returns
        -------
        dict[str, ndarray]
            Mapping from variable name to (n,) array of sampled values.
        """
        rng = np.random.default_rng(seed)
        if interventions is None:
            interventions = {}

        data: Dict[str, NDArray[np.floating]] = {}
        for var in self.topological_order:
            if var in interventions:
                data[var] = np.full(n, interventions[var], dtype=np.float64)
                continue

            eq = self.equations.get(var)
            if eq is None:
                # Exogenous root – draw from noise distribution
                dist = self.exogenous_distributions.get(var)
                if dist is not None:
                    data[var] = dist.rvs(size=n, random_state=rng)
                else:
                    data[var] = rng.normal(size=n)
                continue

            noise_dist = self.exogenous_distributions.get(var)
            if noise_dist is not None:
                noises = noise_dist.rvs(size=n, random_state=rng)
            else:
                noises = rng.normal(scale=getattr(eq, "noise_std", 1.0), size=n)

            vals = np.empty(n, dtype=np.float64)
            for i in range(n):
                parent_vals = {p: float(data[p][i]) for p in self.parents(var)}
                vals[i] = eq.evaluate(parent_vals, noise=float(noises[i]))
            data[var] = vals

        return data

    def sample_dataframe(
        self,
        n: int = 1000,
        interventions: Optional[Dict[str, float]] = None,
        seed: Optional[int] = None,
    ) -> Any:
        """Forward-sample and return a pandas DataFrame (if pandas available)."""
        data = self.sample(n=n, interventions=interventions, seed=seed)
        try:
            import pandas as pd
            return pd.DataFrame(data)
        except ImportError:
            return data

    # ------------------------------------------------------- interventions
    def do(
        self, interventions: Dict[str, float]
    ) -> "StructuralCausalModel":
        """Return a *mutilated* SCM corresponding to ``do(X=x)``.

        In the mutilated model every intervened variable has its incoming
        edges removed and its structural equation replaced with a constant.
        """
        mutilated = copy.deepcopy(self)
        for var, val in interventions.items():
            if var not in mutilated.graph:
                raise ValueError(f"Variable {var} not in SCM.")
            # Remove incoming edges
            for parent in list(mutilated.graph.predecessors(var)):
                mutilated.graph.remove_edge(parent, var)
            # Replace equation with constant
            mutilated.equations[var] = LinearEquation(
                weights={}, intercept=val, noise_std=0.0
            )
        mutilated._topological_order = None
        return mutilated

    def interventional_distribution(
        self,
        interventions: Dict[str, float],
        target: str,
        n: int = 5000,
        seed: Optional[int] = None,
    ) -> NDArray[np.floating]:
        """Sample the interventional distribution P(target | do(interventions))."""
        mutilated = self.do(interventions)
        samples = mutilated.sample(n=n, seed=seed)
        return samples[target]

    # ------------------------------------------------------ counterfactuals
    def counterfactual(
        self,
        factual_observation: Dict[str, float],
        intervention: Dict[str, float],
        target: str,
    ) -> float:
        """Three-step counterfactual computation (abduction-action-prediction).

        1. **Abduction**: infer exogenous noise values consistent with the
           factual observation.
        2. **Action**: apply ``do(intervention)`` to get the mutilated model.
        3. **Prediction**: forward-compute the target variable under the
           inferred noise.

        Currently supports only models with ``LinearEquation`` (closed-form
        noise recovery).
        """
        # Step 1 – Abduction: recover noise terms
        noises: Dict[str, float] = {}
        for var in self.topological_order:
            obs_val = factual_observation.get(var)
            if obs_val is None:
                noises[var] = 0.0
                continue
            eq = self.equations.get(var)
            if eq is None:
                noises[var] = obs_val
                continue
            parent_vals = {
                p: factual_observation.get(p, 0.0) for p in self.parents(var)
            }
            deterministic_val = eq.evaluate(parent_vals, noise=0.0)
            noises[var] = obs_val - deterministic_val

        # Step 2 – Action: mutilate
        mutilated = self.do(intervention)

        # Step 3 – Prediction: forward-compute with recovered noise
        values: Dict[str, float] = {}
        for var in mutilated.topological_order:
            if var in intervention:
                values[var] = intervention[var]
                continue
            eq = mutilated.equations.get(var)
            if eq is None:
                values[var] = noises.get(var, 0.0)
                continue
            parent_vals = {p: values.get(p, 0.0) for p in mutilated.parents(var)}
            values[var] = eq.evaluate(parent_vals, noise=noises.get(var, 0.0))

        return values[target]

    def counterfactual_distribution(
        self,
        factual_observations: Dict[str, NDArray[np.floating]],
        intervention: Dict[str, float],
        target: str,
    ) -> NDArray[np.floating]:
        """Vectorised counterfactual over many factual samples."""
        n = len(next(iter(factual_observations.values())))
        results = np.empty(n)
        for i in range(n):
            obs = {k: float(v[i]) for k, v in factual_observations.items()}
            results[i] = self.counterfactual(obs, intervention, target)
        return results

    # --------------------------------------------------- conditional independ
    def implied_independencies(self) -> List[Tuple[str, str, FrozenSet[str]]]:
        """Enumerate pairwise conditional independencies implied by the DAG
        (up to conditioning sets of bounded size)."""
        nodes = self.variables
        independencies: List[Tuple[str, str, FrozenSet[str]]] = []
        other_nodes = set(nodes)
        for i, x in enumerate(nodes):
            for y in nodes[i + 1:]:
                remaining = other_nodes - {x, y}
                for size in range(len(remaining) + 1):
                    for subset in itertools.combinations(remaining, size):
                        z = frozenset(subset)
                        if self.d_separated(x, y, set(z)):
                            independencies.append((x, y, z))
                            break  # smallest separating set is enough
        return independencies

    # ------------------------------------------------- causal effect (linear)
    def total_causal_effect_linear(
        self, cause: str, effect: str
    ) -> float:
        """Compute total causal effect in a linear SCM by summing over all
        directed paths from *cause* to *effect*."""
        if cause == effect:
            return 1.0
        total = 0.0
        for path in nx.all_simple_paths(self.graph, cause, effect):
            path_weight = 1.0
            for u, v in zip(path[:-1], path[1:]):
                eq = self.equations.get(v)
                if isinstance(eq, LinearEquation):
                    w = eq.weights.get(u, 0.0)
                else:
                    w = 0.0
                path_weight *= w
            total += path_weight
        return total

    # ----------------------------------------------------- adjustment sets
    def valid_adjustment_set(
        self,
        treatment: str,
        outcome: str,
        candidate: Set[str],
    ) -> bool:
        """Check the back-door criterion for a candidate adjustment set.

        The back-door criterion holds iff:
        1. No node in *candidate* is a descendant of *treatment*.
        2. *candidate* blocks every path between *treatment* and *outcome*
           that contains an arrow into *treatment* (back-door paths).
        """
        descendants = self.descendants(treatment)
        if candidate & descendants:
            return False

        # Remove outgoing edges from treatment, then check d-separation
        temp_graph = self.graph.copy()
        for child in list(temp_graph.successors(treatment)):
            temp_graph.remove_edge(treatment, child)

        temp_scm = StructuralCausalModel("temp")
        temp_scm.graph = temp_graph

        return temp_scm.d_separated(treatment, outcome, candidate)

    def find_minimal_adjustment_set(
        self, treatment: str, outcome: str
    ) -> Optional[Set[str]]:
        """Find a minimal valid adjustment set (smallest set satisfying the
        back-door criterion), via brute-force search."""
        candidates = set(self.variables) - {treatment, outcome}
        descendants_t = self.descendants(treatment)
        non_desc = candidates - descendants_t

        for size in range(len(non_desc) + 1):
            for subset in itertools.combinations(non_desc, size):
                s = set(subset)
                if self.valid_adjustment_set(treatment, outcome, s):
                    return s
        return None

    # ------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        """Serialise the SCM to a JSON-compatible dictionary."""
        eq_dicts: Dict[str, Any] = {}
        for var, eq in self.equations.items():
            eq_dicts[var] = eq.to_dict()

        return {
            "name": self.name,
            "nodes": list(self.graph.nodes),
            "edges": list(self.graph.edges),
            "equations": eq_dicts,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "StructuralCausalModel":
        """Reconstruct an SCM from a serialised dictionary."""
        scm = cls(name=d.get("name", "SCM"))
        for node in d["nodes"]:
            scm.graph.add_node(node)
        for u, v in d["edges"]:
            scm.graph.add_edge(u, v)

        eq_registry = {"linear": LinearEquation, "anm": ANMEquation}
        for var, eq_d in d.get("equations", {}).items():
            eq_cls = eq_registry.get(eq_d["type"])
            if eq_cls is not None:
                scm.equations[var] = eq_cls.from_dict(eq_d)

        scm._topological_order = None
        return scm

    def to_json(self, path: Optional[str] = None) -> str:
        """Serialise to JSON string; optionally write to *path*."""
        s = json.dumps(self.to_dict(), indent=2)
        if path is not None:
            with open(path, "w") as fh:
                fh.write(s)
        return s

    @classmethod
    def from_json(cls, source: str) -> "StructuralCausalModel":
        """Load from a JSON string or file path."""
        try:
            d = json.loads(source)
        except json.JSONDecodeError:
            with open(source) as fh:
                d = json.load(fh)
        return cls.from_dict(d)

    # ---------------------------------------------------------- utilities
    def copy(self) -> "StructuralCausalModel":
        return copy.deepcopy(self)

    def __repr__(self) -> str:
        return (
            f"StructuralCausalModel(name={self.name!r}, "
            f"variables={len(self.variables)}, edges={len(self.edges)})"
        )

    # ------------------------------------------------------- random factory
    @staticmethod
    def random_linear(
        n_variables: int = 5,
        edge_prob: float = 0.3,
        weight_range: Tuple[float, float] = (-2.0, 2.0),
        noise_std: float = 1.0,
        seed: Optional[int] = None,
    ) -> "StructuralCausalModel":
        """Generate a random linear-Gaussian SCM.

        Constructs an Erdős-Rényi DAG (upper-triangular adjacency) with
        uniformly sampled edge weights.
        """
        rng = np.random.default_rng(seed)
        names = [f"X{i}" for i in range(n_variables)]
        scm = StructuralCausalModel(name="random_linear")
        for name in names:
            scm.graph.add_node(name)

        perm = rng.permutation(n_variables)

        for i in range(n_variables):
            weights: Dict[str, float] = {}
            for j in range(i):
                if rng.random() < edge_prob:
                    parent = names[int(perm[j])]
                    child = names[int(perm[i])]
                    w = rng.uniform(*weight_range)
                    scm.graph.add_edge(parent, child)
                    weights[parent] = float(w)
            child_name = names[int(perm[i])]
            scm.equations[child_name] = LinearEquation(
                weights=weights, intercept=0.0, noise_std=noise_std
            )
            scm.exogenous_distributions[child_name] = stats.norm(0, noise_std)

        scm._topological_order = None
        return scm

    @staticmethod
    def random_anm(
        n_variables: int = 5,
        edge_prob: float = 0.3,
        noise_std: float = 0.5,
        seed: Optional[int] = None,
    ) -> "StructuralCausalModel":
        """Generate a random additive-noise SCM with nonlinear equations."""
        rng = np.random.default_rng(seed)
        names = [f"X{i}" for i in range(n_variables)]
        scm = StructuralCausalModel(name="random_anm")
        for name in names:
            scm.graph.add_node(name)

        perm = rng.permutation(n_variables)
        func_choices = [
            ("tanh", lambda *a: np.tanh(sum(a))),
            ("sin", lambda *a: np.sin(sum(a))),
            ("cube", lambda *a: sum(a) ** 3),
            ("sigmoid", lambda *a: 1.0 / (1.0 + np.exp(-sum(a)))),
            ("abs", lambda *a: np.abs(sum(a))),
        ]

        for i in range(n_variables):
            parent_names: List[str] = []
            for j in range(i):
                if rng.random() < edge_prob:
                    parent = names[int(perm[j])]
                    child = names[int(perm[i])]
                    scm.graph.add_edge(parent, child)
                    parent_names.append(parent)

            child_name = names[int(perm[i])]
            if parent_names:
                desc, func = func_choices[int(rng.integers(len(func_choices)))]
                scm.equations[child_name] = ANMEquation(
                    func=func,
                    parent_names=parent_names,
                    noise_std=noise_std,
                    description=desc,
                )
            scm.exogenous_distributions[child_name] = stats.norm(0, noise_std)

        scm._topological_order = None
        return scm
