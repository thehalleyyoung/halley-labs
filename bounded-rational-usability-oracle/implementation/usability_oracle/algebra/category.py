"""
usability_oracle.algebra.category — Category-theoretic cost algebra.

Provides a monoidal category structure for compositional reasoning about
cognitive costs in UI task graphs.

Mathematical Structure
----------------------
We define a **symmetric monoidal category** ``CostCat``:

* **Objects**: cognitive states (described by a label and capacity vector)
* **Morphisms**: cost-annotated transitions between states, carrying a
  :class:`CostElement` payload
* **Composition (;)**: sequential composition of morphisms via ⊕
* **Tensor product (⊗)**: parallel composition of morphisms
* **Unit object (I)**: idle cognitive state
* **Coherence**: associator α, left/right unitors λ/ρ, braiding σ

Applications
~~~~~~~~~~~~
* **String diagram interpretation**: compose morphisms by wiring diagrams
* **Functor from task graph**: map a DAG of UI tasks to morphisms in CostCat
* **Natural transformations**: parameter changes induce natural transformations
  between cost functors

References
----------
* Mac Lane, *Categories for the Working Mathematician*, Ch. VII (monoidal)
* Fong & Spivak, *An Invitation to Applied Category Theory* (string diagrams)
* Piedeleu & Zanasi, *An Introduction to String Diagrams* (2023)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import networkx as nx

from usability_oracle.algebra.models import CostElement
from usability_oracle.algebra.sequential import SequentialComposer
from usability_oracle.algebra.parallel import ParallelComposer

# ---------------------------------------------------------------------------
# Cognitive state (objects of the category)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CognitiveState:
    """An object in the cost category — a cognitive state.

    Parameters
    ----------
    label : str
        Human-readable identifier (e.g., ``"idle"``, ``"focused"``, ``"fatigued"``).
    capacity : tuple[float, ...]
        Resource capacity vector (one entry per Wickens resource channel).
        Convention: 8-dimensional for the MRT channels.
    """

    label: str
    capacity: Tuple[float, ...] = (1.0,) * 8

    @classmethod
    def idle(cls) -> "CognitiveState":
        """The monoidal unit object — full capacity, no load."""
        return cls(label="idle", capacity=(1.0,) * 8)

    @classmethod
    def from_load(cls, label: str, loads: Sequence[float]) -> "CognitiveState":
        """Create a state with remaining capacity ``1 - load`` per channel."""
        cap = tuple(max(0.0, 1.0 - l) for l in loads)
        return cls(label=label, capacity=cap)

    def total_capacity(self) -> float:
        """Sum of remaining capacity across all channels."""
        return sum(self.capacity)

    def __repr__(self) -> str:
        return f"CogState({self.label!r})"


# ---------------------------------------------------------------------------
# Morphism (arrows of the category)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CostMorphism:
    r"""A morphism in the cost category — a cost-annotated cognitive transition.

    A morphism ``f : A → B`` carries a :class:`CostElement` describing the
    cognitive cost of transitioning from state *A* to state *B*.

    Parameters
    ----------
    source : CognitiveState
        Domain of the morphism.
    target : CognitiveState
        Codomain of the morphism.
    cost : CostElement
        Cognitive cost of the transition.
    label : str
        Descriptive label (e.g., the name of the UI action).
    """

    source: CognitiveState
    target: CognitiveState
    cost: CostElement
    label: str = ""

    # -- identity morphism ---------------------------------------------------

    @classmethod
    def identity(cls, state: CognitiveState) -> "CostMorphism":
        """Identity morphism ``id_A : A → A`` with zero cost."""
        return cls(source=state, target=state, cost=CostElement.zero(), label="id")

    # -- introspection -------------------------------------------------------

    def is_identity(self, tol: float = 1e-12) -> bool:
        """True if this morphism is (approximately) an identity."""
        return (
            self.source == self.target
            and abs(self.cost.mu) < tol
            and abs(self.cost.sigma_sq) < tol
        )

    def __repr__(self) -> str:
        return (
            f"CostMorphism({self.source.label}→{self.target.label}, "
            f"μ={self.cost.mu:.4f}, label={self.label!r})"
        )


# ---------------------------------------------------------------------------
# CostCategory — the monoidal category
# ---------------------------------------------------------------------------


class CostCategory:
    """A symmetric monoidal category of cognitive states and cost transitions.

    Composition is sequential (;) via :class:`SequentialComposer`;
    tensor product is parallel (⊗) via :class:`ParallelComposer`.

    This class enforces typing (source/target matching) and provides
    the coherence isomorphisms required by the monoidal axioms.

    Parameters
    ----------
    default_coupling : float
        Default coupling for sequential composition.
    default_interference : float
        Default interference for parallel composition.
    """

    def __init__(
        self,
        default_coupling: float = 0.0,
        default_interference: float = 0.0,
    ) -> None:
        self._seq = SequentialComposer()
        self._par = ParallelComposer()
        self._coupling = default_coupling
        self._interference = default_interference
        self._unit = CognitiveState.idle()

    @property
    def unit(self) -> CognitiveState:
        """The monoidal unit object I."""
        return self._unit

    # -- sequential composition (;) ------------------------------------------

    def compose(
        self,
        f: CostMorphism,
        g: CostMorphism,
        coupling: Optional[float] = None,
    ) -> CostMorphism:
        r"""Compose morphisms sequentially: ``g ∘ f : A → C``.

        Requires ``f.target == g.source`` (type safety).

        Parameters
        ----------
        f : CostMorphism
            First morphism ``f : A → B``.
        g : CostMorphism
            Second morphism ``g : B → C``.
        coupling : float | None
            Override for coupling parameter.

        Returns
        -------
        CostMorphism
            The composition ``g ∘ f : A → C``.

        Raises
        ------
        TypeError
            If ``f.target != g.source``.
        """
        if f.target != g.source:
            raise TypeError(
                f"Cannot compose: f.target={f.target} != g.source={g.source}"
            )
        rho = coupling if coupling is not None else self._coupling
        cost = self._seq.compose(f.cost, g.cost, coupling=rho)
        return CostMorphism(
            source=f.source,
            target=g.target,
            cost=cost,
            label=f"{f.label};{g.label}",
        )

    def compose_chain(
        self,
        morphisms: List[CostMorphism],
        coupling: Optional[float] = None,
    ) -> CostMorphism:
        """Compose a chain of morphisms: ``fₙ ∘ … ∘ f₂ ∘ f₁``.

        Parameters
        ----------
        morphisms : list[CostMorphism]
            Morphisms to compose, in execution order.

        Returns
        -------
        CostMorphism
        """
        if not morphisms:
            return CostMorphism.identity(self._unit)
        result = morphisms[0]
        for m in morphisms[1:]:
            result = self.compose(result, m, coupling=coupling)
        return result

    # -- tensor product (⊗) --------------------------------------------------

    def tensor(
        self,
        f: CostMorphism,
        g: CostMorphism,
        interference: Optional[float] = None,
    ) -> CostMorphism:
        r"""Tensor product of morphisms: ``f ⊗ g : A⊗C → B⊗D``.

        Represents parallel execution of two transitions.

        Parameters
        ----------
        f : CostMorphism
            First morphism ``f : A → B``.
        g : CostMorphism
            Second morphism ``g : C → D``.
        interference : float | None
            Override for interference parameter.

        Returns
        -------
        CostMorphism
            The tensor product ``f ⊗ g``.
        """
        eta = interference if interference is not None else self._interference
        cost = self._par.compose(f.cost, g.cost, interference=eta)
        src = self._tensor_state(f.source, g.source)
        tgt = self._tensor_state(f.target, g.target)
        return CostMorphism(
            source=src,
            target=tgt,
            cost=cost,
            label=f"({f.label}⊗{g.label})",
        )

    def tensor_many(
        self,
        morphisms: List[CostMorphism],
        interference: Optional[float] = None,
    ) -> CostMorphism:
        """N-ary tensor product via left fold."""
        if not morphisms:
            return CostMorphism.identity(self._unit)
        result = morphisms[0]
        for m in morphisms[1:]:
            result = self.tensor(result, m, interference=interference)
        return result

    # -- coherence isomorphisms ----------------------------------------------

    def associator(
        self, a: CognitiveState, b: CognitiveState, c: CognitiveState
    ) -> CostMorphism:
        r"""Associator ``α_{A,B,C} : (A⊗B)⊗C → A⊗(B⊗C)``.

        An isomorphism (zero cost) witnessing that tensor is associative
        up to coherent natural isomorphism.
        """
        src = self._tensor_state(self._tensor_state(a, b), c)
        tgt = self._tensor_state(a, self._tensor_state(b, c))
        return CostMorphism(source=src, target=tgt, cost=CostElement.zero(), label="α")

    def left_unitor(self, a: CognitiveState) -> CostMorphism:
        r"""Left unitor ``λ_A : I ⊗ A → A``."""
        src = self._tensor_state(self._unit, a)
        return CostMorphism(source=src, target=a, cost=CostElement.zero(), label="λ")

    def right_unitor(self, a: CognitiveState) -> CostMorphism:
        r"""Right unitor ``ρ_A : A ⊗ I → A``."""
        src = self._tensor_state(a, self._unit)
        return CostMorphism(source=src, target=a, cost=CostElement.zero(), label="ρ")

    def braiding(self, a: CognitiveState, b: CognitiveState) -> CostMorphism:
        r"""Braiding (swap) ``σ_{A,B} : A ⊗ B → B ⊗ A``.

        The symmetric structure: parallel composition is commutative.
        The braiding carries zero cost because ⊗ is commutative in CostCat.
        """
        src = self._tensor_state(a, b)
        tgt = self._tensor_state(b, a)
        return CostMorphism(source=src, target=tgt, cost=CostElement.zero(), label="σ")

    # -- verification of monoidal axioms -------------------------------------

    def verify_pentagon(
        self,
        a: CognitiveState,
        b: CognitiveState,
        c: CognitiveState,
        d: CognitiveState,
        tol: float = 1e-10,
    ) -> bool:
        r"""Verify the pentagon identity for the associator.

        The two paths from ``((A⊗B)⊗C)⊗D`` to ``A⊗(B⊗(C⊗D))``
        must agree (both have zero cost in CostCat).
        """
        # Path 1: α_{A⊗B,C,D} ; α_{A,B,C⊗D}
        ab = self._tensor_state(a, b)
        cd = self._tensor_state(c, d)
        alpha1 = self.associator(ab, c, d)
        alpha2 = self.associator(a, b, cd)
        path1_cost = self._seq.compose(alpha1.cost, alpha2.cost)

        # Path 2: (α_{A,B,C} ⊗ id_D) ; α_{A,B⊗C,D} ; (id_A ⊗ α_{B,C,D})
        bc = self._tensor_state(b, c)
        alpha3 = self.associator(a, b, c)
        id_d = CostMorphism.identity(d)
        step1 = self.tensor(alpha3, id_d)

        alpha4 = self.associator(a, bc, d)
        step12 = self._seq.compose(step1.cost, alpha4.cost)

        alpha5 = self.associator(b, c, d)
        id_a = CostMorphism.identity(a)
        step3 = self.tensor(id_a, alpha5)
        path2_cost = self._seq.compose(
            CostElement(mu=step12.mu, sigma_sq=step12.sigma_sq,
                        kappa=step12.kappa, lambda_=step12.lambda_),
            step3.cost,
        )

        return (
            abs(path1_cost.mu - path2_cost.mu) < tol
            and abs(path1_cost.sigma_sq - path2_cost.sigma_sq) < tol
        )

    def verify_triangle(
        self, a: CognitiveState, b: CognitiveState, tol: float = 1e-10
    ) -> bool:
        r"""Verify the triangle identity: ``α_{A,I,B} ; (id_A ⊗ λ_B) = ρ_A ⊗ id_B``.

        Both sides must equal zero cost.
        """
        alpha = self.associator(a, self._unit, b)
        lam = self.left_unitor(b)
        id_a = CostMorphism.identity(a)
        lhs = self._seq.compose(alpha.cost, self.tensor(id_a, lam).cost)

        rho = self.right_unitor(a)
        id_b = CostMorphism.identity(b)
        rhs = self.tensor(rho, id_b).cost

        return abs(lhs.mu - rhs.mu) < tol and abs(lhs.sigma_sq - rhs.sigma_sq) < tol

    def verify_hexagon(
        self, a: CognitiveState, b: CognitiveState, c: CognitiveState,
        tol: float = 1e-10,
    ) -> bool:
        r"""Verify the hexagon identity for the braiding (symmetric case).

        Both paths from ``(A⊗B)⊗C`` to ``(B⊗C)⊗A`` should yield zero cost.
        """
        # Path 1: α ; σ ; α
        step1 = self.associator(a, b, c)
        bc = self._tensor_state(b, c)
        step2 = self.braiding(a, bc)
        step3 = self.associator(b, c, a)
        path1 = self._seq.compose(
            self._seq.compose(step1.cost, step2.cost), step3.cost
        )

        # Path 2: (σ ⊗ id) ; α ; (id ⊗ σ)
        swap_ab = self.braiding(a, b)
        id_c = CostMorphism.identity(c)
        s1 = self.tensor(swap_ab, id_c)
        s2 = self.associator(b, a, c)
        id_b = CostMorphism.identity(b)
        swap_ac = self.braiding(a, c)
        s3 = self.tensor(id_b, swap_ac)
        path2 = self._seq.compose(
            self._seq.compose(s1.cost, s2.cost), s3.cost
        )

        return abs(path1.mu - path2.mu) < tol and abs(path1.sigma_sq - path2.sigma_sq) < tol

    # -- string diagram interpretation ---------------------------------------

    def interpret_string_diagram(
        self,
        wires: List[CognitiveState],
        boxes: List[Tuple[str, List[int], List[int], CostMorphism]],
    ) -> CostMorphism:
        r"""Interpret a string diagram as a composite morphism.

        A string diagram is specified by:
        * ``wires``: a list of cognitive states labelling the wires.
        * ``boxes``: a list of ``(name, input_wires, output_wires, morphism)``
          tuples, given in topological order (left to right in the diagram).

        The boxes are composed sequentially; within each topological level,
        independent boxes are tensored.

        Parameters
        ----------
        wires : list[CognitiveState]
            Wire labels.
        boxes : list[tuple[str, list[int], list[int], CostMorphism]]
            ``(name, [input wire indices], [output wire indices], morphism)``.

        Returns
        -------
        CostMorphism
            The composite morphism.
        """
        if not boxes:
            if wires:
                return CostMorphism.identity(wires[0])
            return CostMorphism.identity(self._unit)

        # Group boxes into sequential layers by dependency
        layers = self._topological_layers(boxes)
        result: Optional[CostMorphism] = None

        for layer in layers:
            layer_morphisms = [box[3] for box in layer]
            if len(layer_morphisms) == 1:
                layer_result = layer_morphisms[0]
            else:
                layer_result = self.tensor_many(layer_morphisms)

            if result is None:
                result = layer_result
            else:
                # For sequential composition, we relax type-checking since
                # the string diagram guarantees consistency by construction.
                cost = self._seq.compose(result.cost, layer_result.cost)
                result = CostMorphism(
                    source=result.source,
                    target=layer_result.target,
                    cost=cost,
                    label=f"{result.label};{layer_result.label}",
                )

        return result if result is not None else CostMorphism.identity(self._unit)

    # -- functor from task graph to CostCat ----------------------------------

    def functor_from_task_graph(
        self,
        graph: nx.DiGraph,
        cost_map: Dict[str, CostElement],
        state_map: Optional[Dict[str, CognitiveState]] = None,
    ) -> CostMorphism:
        r"""Map a task dependency graph to a morphism in CostCat.

        This defines a functor ``F : TaskGraph → CostCat`` that:
        * Maps each task node to a morphism with its associated cost.
        * Maps edges (dependencies) to sequential composition.
        * Maps independent tasks at the same level to parallel composition (tensor).

        Parameters
        ----------
        graph : nx.DiGraph
            Task dependency graph (must be a DAG).
        cost_map : dict[str, CostElement]
            Map from node labels to cost elements.
        state_map : dict[str, CognitiveState] | None
            Optional map from node labels to cognitive states.
            If ``None``, default states are generated.

        Returns
        -------
        CostMorphism
            The composite morphism representing the entire task graph.
        """
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Task graph must be a DAG.")

        if state_map is None:
            state_map = {
                node: CognitiveState(label=str(node)) for node in graph.nodes()
            }

        if len(graph.nodes()) == 0:
            return CostMorphism.identity(self._unit)

        # Topological sort into levels
        levels = self._graph_levels(graph)

        result: Optional[CostMorphism] = None
        for level in levels:
            morphisms = []
            for node in level:
                cost = cost_map.get(str(node), CostElement.zero())
                src = state_map.get(str(node), CognitiveState(label=f"pre_{node}"))
                tgt = state_map.get(str(node), CognitiveState(label=f"post_{node}"))
                if src == tgt:
                    tgt = CognitiveState(label=f"post_{node}", capacity=src.capacity)
                morphisms.append(
                    CostMorphism(source=src, target=tgt, cost=cost, label=str(node))
                )

            if len(morphisms) == 1:
                level_m = morphisms[0]
            else:
                level_m = self.tensor_many(morphisms)

            if result is None:
                result = level_m
            else:
                cost = self._seq.compose(result.cost, level_m.cost, coupling=self._coupling)
                result = CostMorphism(
                    source=result.source,
                    target=level_m.target,
                    cost=cost,
                    label=f"{result.label};{level_m.label}",
                )

        return result if result is not None else CostMorphism.identity(self._unit)

    # -- natural transformations ---------------------------------------------

    def natural_transformation(
        self,
        graph: nx.DiGraph,
        cost_map_1: Dict[str, CostElement],
        cost_map_2: Dict[str, CostElement],
    ) -> Dict[str, CostElement]:
        r"""Compute the components of a natural transformation between two
        cost functors on the same task graph.

        Given two cost assignments ``C₁, C₂`` on the same graph, a natural
        transformation ``η : F₁ ⇒ F₂`` has components ``η_v = C₂(v) - C₁(v)``
        at each node ``v``.  The *naturality condition* requires that for
        each edge ``u → v``:

        .. math::

            F₂(u→v) ∘ η_u = η_v ∘ F₁(u→v)

        In our setting (additive costs), this reduces to checking that the
        difference is consistent under composition.

        Parameters
        ----------
        graph : nx.DiGraph
            Task graph.
        cost_map_1, cost_map_2 : dict[str, CostElement]
            Two cost assignments.

        Returns
        -------
        dict[str, CostElement]
            Components ``η_v`` for each node ``v``.
        """
        components: Dict[str, CostElement] = {}
        for node in graph.nodes():
            c1 = cost_map_1.get(str(node), CostElement.zero())
            c2 = cost_map_2.get(str(node), CostElement.zero())
            components[str(node)] = CostElement(
                mu=c2.mu - c1.mu,
                sigma_sq=abs(c2.sigma_sq - c1.sigma_sq),
                kappa=c2.kappa - c1.kappa,
                lambda_=max(0.0, min(1.0, c2.lambda_ - c1.lambda_)),
            )
        return components

    def verify_naturality(
        self,
        graph: nx.DiGraph,
        cost_map_1: Dict[str, CostElement],
        cost_map_2: Dict[str, CostElement],
        tol: float = 1e-6,
    ) -> bool:
        r"""Verify the naturality condition for the induced transformation.

        For every edge ``u → v``, checks:

        .. math::

            C_2(u) + C_1(v) \approx C_1(u) + C_2(v)

        (commutativity of the naturality square in the additive setting).
        """
        for u, v in graph.edges():
            c1u = cost_map_1.get(str(u), CostElement.zero())
            c1v = cost_map_1.get(str(v), CostElement.zero())
            c2u = cost_map_2.get(str(u), CostElement.zero())
            c2v = cost_map_2.get(str(v), CostElement.zero())

            lhs_mu = c2u.mu + c1v.mu
            rhs_mu = c1u.mu + c2v.mu
            if abs(lhs_mu - rhs_mu) > tol:
                return False
        return True

    # -- internal helpers ----------------------------------------------------

    @staticmethod
    def _tensor_state(a: CognitiveState, b: CognitiveState) -> CognitiveState:
        """Tensor product of cognitive states: element-wise minimum of capacities."""
        cap = tuple(min(ca, cb) for ca, cb in zip(a.capacity, b.capacity))
        return CognitiveState(label=f"({a.label}⊗{b.label})", capacity=cap)

    @staticmethod
    def _topological_layers(
        boxes: List[Tuple[str, List[int], List[int], CostMorphism]],
    ) -> List[List[Tuple[str, List[int], List[int], CostMorphism]]]:
        """Partition boxes into sequential layers based on wire dependencies."""
        # Track which wires have been produced
        produced: Set[int] = set()
        layers: List[List[Tuple[str, List[int], List[int], CostMorphism]]] = []
        remaining = list(boxes)

        while remaining:
            layer = []
            next_remaining = []
            for box in remaining:
                _, inputs, outputs, _ = box
                if all(w in produced for w in inputs) or not inputs:
                    layer.append(box)
                else:
                    next_remaining.append(box)
            if not layer:
                # Avoid infinite loop: flush everything
                layer = next_remaining
                next_remaining = []
            for box in layer:
                produced.update(box[2])
            layers.append(layer)
            remaining = next_remaining

        return layers

    @staticmethod
    def _graph_levels(graph: nx.DiGraph) -> List[List[Any]]:
        """Partition a DAG into topological levels."""
        in_degree = dict(graph.in_degree())
        levels: List[List[Any]] = []
        current = [n for n, d in in_degree.items() if d == 0]
        while current:
            levels.append(current)
            next_level: List[Any] = []
            for node in current:
                for succ in graph.successors(node):
                    in_degree[succ] -= 1
                    if in_degree[succ] == 0:
                        next_level.append(succ)
            current = next_level
        return levels
