"""Core type definitions for the Causal-Plasticity Atlas engine.

Provides enum classes for plasticity classification, certificate types,
edge classifications, and change types; and dataclasses for SCM,
Context, MCCM, AlignmentMapping, PlasticityDescriptor, TippingPoint,
RobustnessCertificate, QDGenome, QDArchiveEntry, and CVTCell.
"""

from __future__ import annotations

import enum
import json
import math
from collections import deque
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    FrozenSet,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray


# ===================================================================
# Enumerations
# ===================================================================


class PlasticityClass(enum.Enum):
    """Classification of mechanism plasticity across contexts.

    Attributes
    ----------
    INVARIANT
        Mechanism does not change across contexts.
    PARAMETRIC_PLASTIC
        Structure is preserved but parameters differ.
    STRUCTURAL_PLASTIC
        The parent set changes across contexts.
    EMERGENT
        Mechanism appears only in a subset of contexts.
    MIXED
        Combination of parametric and structural plasticity.
    """

    INVARIANT = "invariant"
    PARAMETRIC_PLASTIC = "parametric_plastic"
    STRUCTURAL_PLASTIC = "structural_plastic"
    EMERGENT = "emergent"
    MIXED = "mixed"

    def __repr__(self) -> str:
        return f"PlasticityClass.{self.name}"


class CertificateType(enum.Enum):
    """Type of robustness certificate.

    Attributes
    ----------
    STRONG_INVARIANCE
        Mechanism is provably invariant across all contexts.
    PARAMETRIC_STABILITY
        Parameters stay within bounded region.
    STRUCTURAL_STABILITY
        Structure is stable with bounded perturbations.
    CANNOT_ISSUE
        Robustness cannot be certified.
    """

    STRONG_INVARIANCE = "strong_invariance"
    PARAMETRIC_STABILITY = "parametric_stability"
    STRUCTURAL_STABILITY = "structural_stability"
    CANNOT_ISSUE = "cannot_issue"

    def __repr__(self) -> str:
        return f"CertificateType.{self.name}"


class EdgeClassification(enum.Enum):
    """Classification of an edge in a pairwise context comparison.

    Attributes
    ----------
    SHARED
        Edge is present in both contexts.
    MODIFIED
        Edge is present in both but with different weight/strength.
    CONTEXT_SPECIFIC_A
        Edge is present only in context A.
    CONTEXT_SPECIFIC_B
        Edge is present only in context B.
    """

    SHARED = "shared"
    MODIFIED = "modified"
    CONTEXT_SPECIFIC_A = "context_specific_a"
    CONTEXT_SPECIFIC_B = "context_specific_b"

    def __repr__(self) -> str:
        return f"EdgeClassification.{self.name}"


class ChangeType(enum.Enum):
    """Type of change detected between contexts.

    Attributes
    ----------
    STRUCTURAL
        The directed graph structure differs.
    PARAMETRIC
        Structure is identical but parameters differ.
    """

    STRUCTURAL = "structural"
    PARAMETRIC = "parametric"

    def __repr__(self) -> str:
        return f"ChangeType.{self.name}"


# ===================================================================
# SCM dataclass (lightweight data holder)
# ===================================================================


@dataclass
class SCM:
    """Structural Causal Model data container.

    Holds the adjacency matrix, regression coefficients, residual
    variances, and variable metadata for a single-context causal model.

    Parameters
    ----------
    adjacency_matrix : np.ndarray
        Binary or weighted adjacency matrix, shape ``(p, p)``.
        Entry ``[i, j] != 0`` means variable *i* is a parent of *j*.
    regression_coefficients : np.ndarray
        Weight matrix, shape ``(p, p)``.  ``[i, j]`` is the coefficient
        of variable *i* in the regression for variable *j*.
    residual_variances : np.ndarray
        Residual variance for each variable, shape ``(p,)``.
    variable_names : list of str
        Names for each variable.
    sample_size : int
        Number of observations used to estimate the model.

    Examples
    --------
    >>> adj = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]])
    >>> scm = SCM(adj, adj * 0.5, np.ones(3), ["X", "Y", "Z"], 100)
    >>> scm.parents(1)
    [0]
    """

    adjacency_matrix: NDArray[np.floating]
    regression_coefficients: NDArray[np.floating]
    residual_variances: NDArray[np.floating]
    variable_names: List[str]
    sample_size: int

    def __post_init__(self) -> None:
        self.adjacency_matrix = np.asarray(
            self.adjacency_matrix, dtype=np.float64
        )
        self.regression_coefficients = np.asarray(
            self.regression_coefficients, dtype=np.float64
        )
        self.residual_variances = np.asarray(
            self.residual_variances, dtype=np.float64
        )
        self.variable_names = list(self.variable_names)

        p = self.adjacency_matrix.shape[0]
        if self.adjacency_matrix.shape != (p, p):
            raise ValueError(
                f"adjacency_matrix must be square, got {self.adjacency_matrix.shape}"
            )
        if self.regression_coefficients.shape != (p, p):
            raise ValueError(
                f"regression_coefficients shape {self.regression_coefficients.shape} "
                f"!= expected ({p}, {p})"
            )
        if self.residual_variances.shape != (p,):
            raise ValueError(
                f"residual_variances shape {self.residual_variances.shape} "
                f"!= expected ({p},)"
            )
        if len(self.variable_names) != p:
            raise ValueError(
                f"variable_names length {len(self.variable_names)} != {p}"
            )
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")
        # Check for duplicate names
        seen: set[str] = set()
        for nm in self.variable_names:
            if nm in seen:
                raise ValueError(f"Duplicate variable name: {nm!r}")
            seen.add(nm)

    @property
    def num_variables(self) -> int:
        """Number of variables in the model."""
        return self.adjacency_matrix.shape[0]

    @property
    def num_edges(self) -> int:
        """Number of directed edges."""
        return int(np.count_nonzero(self.adjacency_matrix))

    def variable_index(self, name: str) -> int:
        """Return the integer index of variable *name*.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        int

        Raises
        ------
        ValueError
            If *name* is not found.
        """
        try:
            return self.variable_names.index(name)
        except ValueError:
            raise ValueError(f"Variable {name!r} not in model") from None

    def parents(self, j: int) -> List[int]:
        """Indices of parents of variable *j*.

        Parameters
        ----------
        j : int
            Variable index.

        Returns
        -------
        list of int
        """
        return list(np.nonzero(self.adjacency_matrix[:, j])[0])

    def children(self, i: int) -> List[int]:
        """Indices of children of variable *i*.

        Parameters
        ----------
        i : int
            Variable index.

        Returns
        -------
        list of int
        """
        return list(np.nonzero(self.adjacency_matrix[i, :])[0])

    def markov_blanket(self, i: int) -> Set[int]:
        """Markov blanket of variable *i*: parents + children + co-parents.

        Parameters
        ----------
        i : int
            Variable index.

        Returns
        -------
        set of int
        """
        pa = set(self.parents(i))
        ch = set(self.children(i))
        co_parents: set[int] = set()
        for c in ch:
            co_parents |= set(self.parents(c))
        mb = pa | ch | co_parents
        mb.discard(i)
        return mb

    def topological_sort(self) -> List[int]:
        """Topological ordering of variables (Kahn's algorithm).

        Returns
        -------
        list of int
            Variable indices in topological order.

        Raises
        ------
        ValueError
            If the graph contains a cycle.
        """
        p = self.num_variables
        binary = (self.adjacency_matrix != 0).astype(int)
        in_deg = binary.sum(axis=0).astype(int)
        queue: deque[int] = deque()
        for i in range(p):
            if in_deg[i] == 0:
                queue.append(i)
        order: list[int] = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for child in range(p):
                if binary[node, child]:
                    in_deg[child] -= 1
                    if in_deg[child] == 0:
                        queue.append(child)
        if len(order) != p:
            raise ValueError("Graph contains a cycle")
        return order

    def is_dag_valid(self) -> bool:
        """Check if the adjacency matrix encodes a valid DAG.

        Returns
        -------
        bool
        """
        try:
            self.topological_sort()
            return True
        except ValueError:
            return False

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "adjacency_matrix": self.adjacency_matrix.tolist(),
            "regression_coefficients": self.regression_coefficients.tolist(),
            "residual_variances": self.residual_variances.tolist(),
            "variable_names": list(self.variable_names),
            "sample_size": self.sample_size,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "SCM":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        SCM
        """
        return cls(
            adjacency_matrix=np.array(d["adjacency_matrix"], dtype=np.float64),
            regression_coefficients=np.array(
                d["regression_coefficients"], dtype=np.float64
            ),
            residual_variances=np.array(d["residual_variances"], dtype=np.float64),
            variable_names=list(d["variable_names"]),
            sample_size=int(d["sample_size"]),
        )

    def __repr__(self) -> str:
        return (
            f"SCM(variables={self.variable_names}, "
            f"edges={self.num_edges}, n={self.sample_size})"
        )


# ===================================================================
# Context
# ===================================================================


@dataclass
class Context:
    """A single observational / experimental context.

    Parameters
    ----------
    id : str
        Unique identifier.
    metadata : dict
        Arbitrary key-value metadata describing the context.
    ordering_value : float, optional
        Scalar for ordering contexts (e.g. time, dosage, temperature).
    """

    id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    ordering_value: Optional[float] = None

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise ValueError("Context id must be a non-empty string")
        if not isinstance(self.metadata, dict):
            raise TypeError(
                f"metadata must be a dict, got {type(self.metadata).__name__}"
            )
        if self.ordering_value is not None:
            if not isinstance(self.ordering_value, (int, float)):
                raise TypeError(
                    f"ordering_value must be numeric, "
                    f"got {type(self.ordering_value).__name__}"
                )
            if math.isnan(self.ordering_value) or math.isinf(self.ordering_value):
                raise ValueError("ordering_value must be finite")

    @property
    def is_ordered(self) -> bool:
        """Whether this context has a defined ordering value."""
        return self.ordering_value is not None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        d: Dict[str, Any] = {"id": self.id, "metadata": dict(self.metadata)}
        if self.ordering_value is not None:
            d["ordering_value"] = self.ordering_value
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Context":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        Context
        """
        return cls(
            id=d["id"],
            metadata=dict(d.get("metadata", {})),
            ordering_value=d.get("ordering_value"),
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Context):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}"]
        if self.ordering_value is not None:
            parts.append(f"order={self.ordering_value}")
        if self.metadata:
            parts.append(f"meta={self.metadata}")
        return f"Context({', '.join(parts)})"


# ===================================================================
# MCCM (Multi-Context Causal Model)
# ===================================================================


@dataclass
class MCCM:
    """Multi-Context Causal Model: collection of context-specific SCMs.

    Parameters
    ----------
    scms : dict
        Mapping ``context_id → SCM``.
    context_space : list of Context
        Ordered list of contexts.
    shared_variables : list of str
        Variables present in all contexts.
    context_specific_variables : dict
        Mapping ``context_id → list of extra variable names``.
    """

    scms: Dict[str, SCM] = field(default_factory=dict)
    context_space: List[Context] = field(default_factory=list)
    shared_variables: List[str] = field(default_factory=list)
    context_specific_variables: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate context_id consistency
        ctx_ids = {c.id for c in self.context_space}
        scm_ids = set(self.scms.keys())
        if scm_ids and ctx_ids and scm_ids != ctx_ids:
            missing_ctx = scm_ids - ctx_ids
            missing_scm = ctx_ids - scm_ids
            parts: list[str] = []
            if missing_ctx:
                parts.append(
                    f"SCM ids without context: {sorted(missing_ctx)}"
                )
            if missing_scm:
                parts.append(
                    f"Context ids without SCM: {sorted(missing_scm)}"
                )
            raise ValueError(
                f"MCCM context/SCM mismatch: {'; '.join(parts)}"
            )

    @property
    def num_contexts(self) -> int:
        """Number of contexts."""
        return len(self.context_space)

    @property
    def context_ids(self) -> List[str]:
        """List of context identifiers."""
        return [c.id for c in self.context_space]

    def get_scm(self, context_id: str) -> SCM:
        """Retrieve the SCM for *context_id*.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        SCM

        Raises
        ------
        KeyError
            If *context_id* is not found.
        """
        if context_id not in self.scms:
            raise KeyError(f"Context {context_id!r} not in MCCM")
        return self.scms[context_id]

    def add_context(
        self,
        context: Context,
        scm: SCM,
        *,
        extra_variables: Optional[List[str]] = None,
    ) -> None:
        """Add a context with its SCM.

        Parameters
        ----------
        context : Context
            Context descriptor.
        scm : SCM
            Structural causal model for this context.
        extra_variables : list of str, optional
            Variables unique to this context.
        """
        if context.id in self.scms:
            raise ValueError(f"Context {context.id!r} already exists")
        self.scms[context.id] = scm
        self.context_space.append(context)
        if extra_variables:
            self.context_specific_variables[context.id] = list(extra_variables)

    def remove_context(self, context_id: str) -> SCM:
        """Remove a context and return its SCM.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        SCM
            The removed SCM.

        Raises
        ------
        KeyError
        """
        if context_id not in self.scms:
            raise KeyError(f"Context {context_id!r} not in MCCM")
        scm = self.scms.pop(context_id)
        self.context_space = [c for c in self.context_space if c.id != context_id]
        self.context_specific_variables.pop(context_id, None)
        return scm

    def validate(self) -> List[str]:
        """Run consistency checks and return a list of warnings.

        Returns
        -------
        list of str
            Warning messages (empty if all checks pass).
        """
        warnings_list: list[str] = []
        for cid, scm in self.scms.items():
            if not scm.is_dag_valid():
                warnings_list.append(f"SCM for context {cid!r} is not a valid DAG")
            if scm.sample_size < 30:
                warnings_list.append(
                    f"Context {cid!r} has small sample size ({scm.sample_size})"
                )
        # Check shared variables are truly shared
        if self.shared_variables and self.scms:
            for var in self.shared_variables:
                for cid, scm in self.scms.items():
                    if var not in scm.variable_names:
                        warnings_list.append(
                            f"Shared variable {var!r} missing from context {cid!r}"
                        )
        return warnings_list

    def variable_union(self) -> Set[str]:
        """Union of all variable names across all contexts.

        Returns
        -------
        set of str
        """
        result: set[str] = set()
        for scm in self.scms.values():
            result |= set(scm.variable_names)
        return result

    def variable_intersection(self) -> Set[str]:
        """Intersection of variable names across all contexts.

        Returns
        -------
        set of str
        """
        if not self.scms:
            return set()
        sets = [set(scm.variable_names) for scm in self.scms.values()]
        return sets[0].intersection(*sets[1:])

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "scms": {k: v.to_dict() for k, v in self.scms.items()},
            "context_space": [c.to_dict() for c in self.context_space],
            "shared_variables": list(self.shared_variables),
            "context_specific_variables": {
                k: list(v) for k, v in self.context_specific_variables.items()
            },
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MCCM":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        MCCM
        """
        return cls(
            scms={k: SCM.from_dict(v) for k, v in d["scms"].items()},
            context_space=[Context.from_dict(c) for c in d["context_space"]],
            shared_variables=list(d.get("shared_variables", [])),
            context_specific_variables={
                k: list(v)
                for k, v in d.get("context_specific_variables", {}).items()
            },
        )

    def __repr__(self) -> str:
        return (
            f"MCCM(contexts={self.num_contexts}, "
            f"shared_vars={len(self.shared_variables)}, "
            f"union_vars={len(self.variable_union()) if self.scms else 0})"
        )


# ===================================================================
# AlignmentMapping
# ===================================================================


@dataclass
class AlignmentMapping:
    """Mapping between two SCMs for structural comparison.

    Parameters
    ----------
    pi : dict
        Variable mapping ``{var_A: var_B}`` aligning context A to B.
    quality_score : float
        Overall alignment quality in [0, 1].
    edge_partition : dict
        Partition of edges into categories:
        ``{"shared": set, "modified": set, "context_specific_a": set,
        "context_specific_b": set}``.
    structural_divergence : float
        Aggregate structural divergence score (>= 0).
    """

    pi: Dict[str, str]
    quality_score: float
    edge_partition: Dict[str, Set[Tuple[str, str]]]
    structural_divergence: float

    def __post_init__(self) -> None:
        if not isinstance(self.pi, dict):
            raise TypeError(f"pi must be a dict, got {type(self.pi).__name__}")
        if not 0.0 <= self.quality_score <= 1.0:
            raise ValueError(
                f"quality_score must be in [0,1], got {self.quality_score}"
            )
        if self.structural_divergence < 0:
            raise ValueError(
                f"structural_divergence must be >= 0, "
                f"got {self.structural_divergence}"
            )
        required_keys = {"shared", "modified", "context_specific_a", "context_specific_b"}
        actual_keys = set(self.edge_partition.keys())
        if actual_keys != required_keys:
            raise ValueError(
                f"edge_partition keys must be {required_keys}, got {actual_keys}"
            )
        for k, v in self.edge_partition.items():
            if not isinstance(v, set):
                self.edge_partition[k] = set(v)

    @property
    def num_shared(self) -> int:
        """Number of shared edges."""
        return len(self.edge_partition["shared"])

    @property
    def num_modified(self) -> int:
        """Number of modified edges."""
        return len(self.edge_partition["modified"])

    @property
    def jaccard_index(self) -> float:
        """Jaccard index of shared edges over total unique edges."""
        all_edges = set()
        for v in self.edge_partition.values():
            all_edges |= v
        if not all_edges:
            return 1.0
        return len(self.edge_partition["shared"]) / len(all_edges)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "pi": dict(self.pi),
            "quality_score": self.quality_score,
            "edge_partition": {
                k: [list(e) for e in v]
                for k, v in self.edge_partition.items()
            },
            "structural_divergence": self.structural_divergence,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "AlignmentMapping":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        AlignmentMapping
        """
        return cls(
            pi=dict(d["pi"]),
            quality_score=float(d["quality_score"]),
            edge_partition={
                k: {tuple(e) for e in v}
                for k, v in d["edge_partition"].items()
            },
            structural_divergence=float(d["structural_divergence"]),
        )

    def __repr__(self) -> str:
        return (
            f"AlignmentMapping(quality={self.quality_score:.3f}, "
            f"shared={self.num_shared}, modified={self.num_modified}, "
            f"divergence={self.structural_divergence:.3f})"
        )


# ===================================================================
# PlasticityDescriptor
# ===================================================================


@dataclass
class PlasticityDescriptor:
    """Four-dimensional plasticity descriptor for a single variable.

    The descriptor vector ψ = (ψ_S, ψ_P, ψ_E, ψ_CS) where each
    component is in [0, 1]:

    - ψ_S: structural plasticity (parent-set changes)
    - ψ_P: parametric plasticity (coefficient changes)
    - ψ_E: emergent plasticity (variable presence changes)
    - ψ_CS: context-specific plasticity (unique mechanisms)

    Parameters
    ----------
    psi_S : float
        Structural plasticity score.
    psi_P : float
        Parametric plasticity score.
    psi_E : float
        Emergent plasticity score.
    psi_CS : float
        Context-specific plasticity score.
    confidence_intervals : dict
        CI for each component, e.g.
        ``{"psi_S": (0.1, 0.3), "psi_P": (0.0, 0.1)}``.
    classification : PlasticityClass
        Assigned plasticity class.
    variable_index : int
        Index of the variable in the original model.
    variable_name : str
        Name of the variable.
    """

    psi_S: float
    psi_P: float
    psi_E: float
    psi_CS: float
    confidence_intervals: Dict[str, Tuple[float, float]]
    classification: PlasticityClass
    variable_index: int
    variable_name: str

    def __post_init__(self) -> None:
        for name, val in [
            ("psi_S", self.psi_S),
            ("psi_P", self.psi_P),
            ("psi_E", self.psi_E),
            ("psi_CS", self.psi_CS),
        ]:
            if not isinstance(val, (int, float)):
                raise TypeError(f"{name} must be numeric, got {type(val).__name__}")
            if val < 0.0 or val > 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {val}")
        if not isinstance(self.classification, PlasticityClass):
            raise TypeError(
                f"classification must be PlasticityClass, "
                f"got {type(self.classification).__name__}"
            )
        if not isinstance(self.variable_name, str):
            raise TypeError("variable_name must be a string")

    @property
    def vector(self) -> NDArray[np.floating]:
        """Return the 4-D descriptor as a numpy array."""
        return np.array(
            [self.psi_S, self.psi_P, self.psi_E, self.psi_CS], dtype=np.float64
        )

    @property
    def magnitude(self) -> float:
        """L2 norm of the descriptor vector."""
        return float(np.linalg.norm(self.vector))

    @property
    def dominant_dimension(self) -> str:
        """Name of the largest descriptor component."""
        names = ["psi_S", "psi_P", "psi_E", "psi_CS"]
        return names[int(np.argmax(self.vector))]

    def distance_to(self, other: "PlasticityDescriptor") -> float:
        """Euclidean distance to another descriptor.

        Parameters
        ----------
        other : PlasticityDescriptor

        Returns
        -------
        float
        """
        return float(np.linalg.norm(self.vector - other.vector))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "psi_S": self.psi_S,
            "psi_P": self.psi_P,
            "psi_E": self.psi_E,
            "psi_CS": self.psi_CS,
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "classification": self.classification.value,
            "variable_index": self.variable_index,
            "variable_name": self.variable_name,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PlasticityDescriptor":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        PlasticityDescriptor
        """
        return cls(
            psi_S=float(d["psi_S"]),
            psi_P=float(d["psi_P"]),
            psi_E=float(d["psi_E"]),
            psi_CS=float(d["psi_CS"]),
            confidence_intervals={
                k: tuple(v) for k, v in d["confidence_intervals"].items()
            },
            classification=PlasticityClass(d["classification"]),
            variable_index=int(d["variable_index"]),
            variable_name=str(d["variable_name"]),
        )

    def __repr__(self) -> str:
        return (
            f"PlasticityDescriptor({self.variable_name}: "
            f"S={self.psi_S:.2f}, P={self.psi_P:.2f}, "
            f"E={self.psi_E:.2f}, CS={self.psi_CS:.2f}, "
            f"class={self.classification.name})"
        )


# ===================================================================
# TippingPoint
# ===================================================================


@dataclass
class TippingPoint:
    """A detected tipping point in the context ordering.

    Parameters
    ----------
    context_location : int
        Index in the ordered context sequence where the tipping point occurs.
    p_value : float
        Statistical significance of the detected change.
    effect_size : float
        Magnitude of the change (>= 0).
    affected_mechanisms : list of str
        Variable names whose mechanisms change at this point.
    change_types : list of ChangeType
        Types of change for each affected mechanism.
    left_segment : list of str
        Context IDs in the segment before the tipping point.
    right_segment : list of str
        Context IDs in the segment after the tipping point.
    """

    context_location: int
    p_value: float
    effect_size: float
    affected_mechanisms: List[str]
    change_types: List[ChangeType]
    left_segment: List[str]
    right_segment: List[str]

    def __post_init__(self) -> None:
        if self.context_location < 0:
            raise ValueError(
                f"context_location must be >= 0, got {self.context_location}"
            )
        if not 0.0 <= self.p_value <= 1.0:
            raise ValueError(f"p_value must be in [0, 1], got {self.p_value}")
        if self.effect_size < 0:
            raise ValueError(f"effect_size must be >= 0, got {self.effect_size}")
        if len(self.affected_mechanisms) != len(self.change_types):
            raise ValueError(
                f"affected_mechanisms and change_types length mismatch: "
                f"{len(self.affected_mechanisms)} vs {len(self.change_types)}"
            )

    @property
    def is_significant(self) -> bool:
        """Whether the tipping point is significant at α = 0.05."""
        return self.p_value < 0.05

    @property
    def num_affected(self) -> int:
        """Number of affected mechanisms."""
        return len(self.affected_mechanisms)

    @property
    def has_structural_change(self) -> bool:
        """Whether any affected mechanism has structural change."""
        return any(ct == ChangeType.STRUCTURAL for ct in self.change_types)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "context_location": self.context_location,
            "p_value": self.p_value,
            "effect_size": self.effect_size,
            "affected_mechanisms": list(self.affected_mechanisms),
            "change_types": [ct.value for ct in self.change_types],
            "left_segment": list(self.left_segment),
            "right_segment": list(self.right_segment),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TippingPoint":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        TippingPoint
        """
        return cls(
            context_location=int(d["context_location"]),
            p_value=float(d["p_value"]),
            effect_size=float(d["effect_size"]),
            affected_mechanisms=list(d["affected_mechanisms"]),
            change_types=[ChangeType(ct) for ct in d["change_types"]],
            left_segment=list(d["left_segment"]),
            right_segment=list(d["right_segment"]),
        )

    def __repr__(self) -> str:
        return (
            f"TippingPoint(loc={self.context_location}, "
            f"p={self.p_value:.4f}, effect={self.effect_size:.3f}, "
            f"mechanisms={self.num_affected})"
        )


# ===================================================================
# RobustnessCertificate
# ===================================================================


@dataclass
class RobustnessCertificate:
    """Certificate attesting to the robustness of a plasticity finding.

    Parameters
    ----------
    type : CertificateType
        Kind of robustness guarantee.
    validity : bool
        Whether the certificate holds.
    max_sqrt_jsd : float
        Maximum sqrt-JSD observed across contexts (>= 0).
    upper_confidence_bound : float
        Upper confidence bound on the divergence metric.
    robustness_margin : float
        Distance from the decision boundary (positive = robust).
    stability_selection_probs : dict
        Selection probabilities from stability selection.
    assumptions : list of str
        Assumptions under which the certificate is valid.
    validity_conditions : list of str
        Conditions that must hold for validity.
    min_sample_size_warning : bool
        Whether any context has fewer samples than recommended.
    """

    type: CertificateType
    validity: bool
    max_sqrt_jsd: float
    upper_confidence_bound: float
    robustness_margin: float
    stability_selection_probs: Dict[str, float]
    assumptions: List[str]
    validity_conditions: List[str]
    min_sample_size_warning: bool

    def __post_init__(self) -> None:
        if not isinstance(self.type, CertificateType):
            raise TypeError(
                f"type must be CertificateType, got {type(self.type).__name__}"
            )
        if self.max_sqrt_jsd < 0:
            raise ValueError(
                f"max_sqrt_jsd must be >= 0, got {self.max_sqrt_jsd}"
            )
        if self.upper_confidence_bound < 0:
            raise ValueError(
                f"upper_confidence_bound must be >= 0, "
                f"got {self.upper_confidence_bound}"
            )

    @property
    def is_strong(self) -> bool:
        """Whether this is a strong invariance certificate."""
        return self.type == CertificateType.STRONG_INVARIANCE and self.validity

    @property
    def summary(self) -> str:
        """One-line summary of the certificate."""
        status = "VALID" if self.validity else "INVALID"
        warn = " [⚠ small n]" if self.min_sample_size_warning else ""
        return (
            f"{self.type.value}: {status}, "
            f"sqrt-JSD≤{self.max_sqrt_jsd:.4f}, "
            f"margin={self.robustness_margin:.4f}{warn}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "type": self.type.value,
            "validity": self.validity,
            "max_sqrt_jsd": self.max_sqrt_jsd,
            "upper_confidence_bound": self.upper_confidence_bound,
            "robustness_margin": self.robustness_margin,
            "stability_selection_probs": dict(self.stability_selection_probs),
            "assumptions": list(self.assumptions),
            "validity_conditions": list(self.validity_conditions),
            "min_sample_size_warning": self.min_sample_size_warning,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RobustnessCertificate":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        RobustnessCertificate
        """
        return cls(
            type=CertificateType(d["type"]),
            validity=bool(d["validity"]),
            max_sqrt_jsd=float(d["max_sqrt_jsd"]),
            upper_confidence_bound=float(d["upper_confidence_bound"]),
            robustness_margin=float(d["robustness_margin"]),
            stability_selection_probs={
                k: float(v) for k, v in d["stability_selection_probs"].items()
            },
            assumptions=list(d["assumptions"]),
            validity_conditions=list(d["validity_conditions"]),
            min_sample_size_warning=bool(d["min_sample_size_warning"]),
        )

    def __repr__(self) -> str:
        return f"RobustnessCertificate({self.summary})"


# ===================================================================
# QDGenome
# ===================================================================


@dataclass
class QDGenome:
    """Genome for Quality-Diversity search over analysis configurations.

    Parameters
    ----------
    context_subset : set of str
        Subset of context IDs to analyse.
    mechanism_subset : set of str
        Subset of variable names (mechanisms) to focus on.
    analysis_params : dict
        Analysis hyperparameters (alpha, n_bootstrap, method, etc.).
    """

    context_subset: Set[str]
    mechanism_subset: Set[str]
    analysis_params: Dict[str, Any]

    def __post_init__(self) -> None:
        self.context_subset = set(self.context_subset)
        self.mechanism_subset = set(self.mechanism_subset)
        if not isinstance(self.analysis_params, dict):
            raise TypeError("analysis_params must be a dict")

    @property
    def size(self) -> int:
        """Total number of selected contexts + mechanisms."""
        return len(self.context_subset) + len(self.mechanism_subset)

    def mutate(
        self,
        all_contexts: Set[str],
        all_mechanisms: Set[str],
        *,
        mutation_rate: float = 0.1,
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Create a mutated copy of this genome.

        Parameters
        ----------
        all_contexts : set of str
            Universe of context IDs.
        all_mechanisms : set of str
            Universe of mechanism names.
        mutation_rate : float
            Probability of flipping each item.
        rng : np.random.Generator, optional
            Random number generator.

        Returns
        -------
        QDGenome
            Mutated copy.
        """
        rng = rng or np.random.default_rng()
        new_ctx = set(self.context_subset)
        for c in all_contexts:
            if rng.random() < mutation_rate:
                if c in new_ctx:
                    if len(new_ctx) > 2:
                        new_ctx.discard(c)
                else:
                    new_ctx.add(c)
        new_mech = set(self.mechanism_subset)
        for m in all_mechanisms:
            if rng.random() < mutation_rate:
                if m in new_mech:
                    if len(new_mech) > 1:
                        new_mech.discard(m)
                else:
                    new_mech.add(m)
        new_params = dict(self.analysis_params)
        for key, val in new_params.items():
            if isinstance(val, float) and rng.random() < mutation_rate:
                new_params[key] = val * rng.lognormal(0, 0.1)
        return QDGenome(new_ctx, new_mech, new_params)

    def crossover(
        self,
        other: "QDGenome",
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> "QDGenome":
        """Uniform crossover with *other* genome.

        Parameters
        ----------
        other : QDGenome
        rng : np.random.Generator, optional

        Returns
        -------
        QDGenome
            Child genome.
        """
        rng = rng or np.random.default_rng()
        all_ctx = self.context_subset | other.context_subset
        child_ctx = {c for c in all_ctx if rng.random() < 0.5}
        if len(child_ctx) < 2:
            child_ctx = set(list(all_ctx)[:2])
        all_mech = self.mechanism_subset | other.mechanism_subset
        child_mech = {m for m in all_mech if rng.random() < 0.5}
        if not child_mech:
            child_mech = {list(all_mech)[0]}
        child_params = {}
        all_keys = set(self.analysis_params) | set(other.analysis_params)
        for k in all_keys:
            if rng.random() < 0.5 and k in self.analysis_params:
                child_params[k] = self.analysis_params[k]
            elif k in other.analysis_params:
                child_params[k] = other.analysis_params[k]
            elif k in self.analysis_params:
                child_params[k] = self.analysis_params[k]
        return QDGenome(child_ctx, child_mech, child_params)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "context_subset": sorted(self.context_subset),
            "mechanism_subset": sorted(self.mechanism_subset),
            "analysis_params": dict(self.analysis_params),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QDGenome":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        QDGenome
        """
        return cls(
            context_subset=set(d["context_subset"]),
            mechanism_subset=set(d["mechanism_subset"]),
            analysis_params=dict(d["analysis_params"]),
        )

    def __repr__(self) -> str:
        return (
            f"QDGenome(ctx={len(self.context_subset)}, "
            f"mech={len(self.mechanism_subset)}, "
            f"params={list(self.analysis_params.keys())})"
        )


# ===================================================================
# QDArchiveEntry
# ===================================================================


@dataclass
class QDArchiveEntry:
    """An entry in the Quality-Diversity MAP-Elites archive.

    Parameters
    ----------
    genome : QDGenome
        The genome that produced this entry.
    behavior_descriptor : np.ndarray
        4-D behavior descriptor (same space as PlasticityDescriptor).
    quality : float
        Fitness / quality score.
    cell_index : int
        Index of the CVT cell this entry occupies.
    """

    genome: QDGenome
    behavior_descriptor: NDArray[np.floating]
    quality: float
    cell_index: int

    def __post_init__(self) -> None:
        self.behavior_descriptor = np.asarray(
            self.behavior_descriptor, dtype=np.float64
        )
        if self.behavior_descriptor.shape != (4,):
            raise ValueError(
                f"behavior_descriptor must be shape (4,), "
                f"got {self.behavior_descriptor.shape}"
            )
        if self.cell_index < 0:
            raise ValueError(f"cell_index must be >= 0, got {self.cell_index}")

    def dominates(self, other: "QDArchiveEntry") -> bool:
        """Check if this entry dominates *other* (same cell, higher quality).

        Parameters
        ----------
        other : QDArchiveEntry

        Returns
        -------
        bool
        """
        return self.cell_index == other.cell_index and self.quality > other.quality

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "genome": self.genome.to_dict(),
            "behavior_descriptor": self.behavior_descriptor.tolist(),
            "quality": self.quality,
            "cell_index": self.cell_index,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "QDArchiveEntry":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        QDArchiveEntry
        """
        return cls(
            genome=QDGenome.from_dict(d["genome"]),
            behavior_descriptor=np.array(d["behavior_descriptor"], dtype=np.float64),
            quality=float(d["quality"]),
            cell_index=int(d["cell_index"]),
        )

    def __repr__(self) -> str:
        bd = self.behavior_descriptor
        return (
            f"QDArchiveEntry(cell={self.cell_index}, "
            f"quality={self.quality:.4f}, "
            f"bd=[{bd[0]:.2f},{bd[1]:.2f},{bd[2]:.2f},{bd[3]:.2f}])"
        )


# ===================================================================
# CVTCell
# ===================================================================


@dataclass
class CVTCell:
    """A cell in the Centroidal Voronoi Tessellation archive.

    Parameters
    ----------
    centroid : np.ndarray
        Cell centroid in behavior space, shape ``(4,)``.
    entries : list of QDArchiveEntry
        Entries assigned to this cell (best-first).
    visit_count : int
        Number of times this cell has been visited.
    quality_ema : float
        Exponential moving average of quality for entries placed here.
    """

    centroid: NDArray[np.floating]
    entries: List[QDArchiveEntry] = field(default_factory=list)
    visit_count: int = 0
    quality_ema: float = 0.0

    def __post_init__(self) -> None:
        self.centroid = np.asarray(self.centroid, dtype=np.float64)
        if self.centroid.shape != (4,):
            raise ValueError(
                f"centroid must be shape (4,), got {self.centroid.shape}"
            )
        if self.visit_count < 0:
            raise ValueError(f"visit_count must be >= 0, got {self.visit_count}")

    @property
    def is_empty(self) -> bool:
        """Whether this cell has no entries."""
        return len(self.entries) == 0

    @property
    def best_quality(self) -> Optional[float]:
        """Highest quality among entries, or None if empty."""
        if not self.entries:
            return None
        return max(e.quality for e in self.entries)

    @property
    def best_entry(self) -> Optional[QDArchiveEntry]:
        """Entry with the highest quality, or None if empty."""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.quality)

    def add_entry(self, entry: QDArchiveEntry, *, ema_alpha: float = 0.1) -> bool:
        """Attempt to add an entry. Returns True if it was added/replaced.

        Parameters
        ----------
        entry : QDArchiveEntry
            Candidate entry.
        ema_alpha : float
            EMA smoothing factor.

        Returns
        -------
        bool
            ``True`` if the entry was kept (new best or cell was empty).
        """
        self.visit_count += 1
        self.quality_ema = (1 - ema_alpha) * self.quality_ema + ema_alpha * entry.quality

        if not self.entries:
            self.entries.append(entry)
            return True
        if entry.quality > self.entries[0].quality:
            self.entries.insert(0, entry)
            return True
        return False

    def distance_to(self, point: NDArray[np.floating]) -> float:
        """Euclidean distance from *point* to this cell's centroid.

        Parameters
        ----------
        point : np.ndarray
            Point in behavior space, shape ``(4,)``.

        Returns
        -------
        float
        """
        return float(np.linalg.norm(self.centroid - np.asarray(point)))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "centroid": self.centroid.tolist(),
            "entries": [e.to_dict() for e in self.entries],
            "visit_count": self.visit_count,
            "quality_ema": self.quality_ema,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CVTCell":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        CVTCell
        """
        return cls(
            centroid=np.array(d["centroid"], dtype=np.float64),
            entries=[QDArchiveEntry.from_dict(e) for e in d["entries"]],
            visit_count=int(d["visit_count"]),
            quality_ema=float(d["quality_ema"]),
        )

    def __repr__(self) -> str:
        c = self.centroid
        best = self.best_quality
        best_str = f"{best:.4f}" if best is not None else "∅"
        return (
            f"CVTCell(centroid=[{c[0]:.2f},{c[1]:.2f},{c[2]:.2f},{c[3]:.2f}], "
            f"entries={len(self.entries)}, visits={self.visit_count}, "
            f"best={best_str})"
        )
