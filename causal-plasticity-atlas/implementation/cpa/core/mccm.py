"""Multi-Context Causal Model (MCCM) for the CPA engine.

Provides the :class:`MultiContextCausalModel` class with full
variable-set mismatch handling, context addition/removal, pairwise
comparison infrastructure, batch operations, summary statistics,
and serialisation.
"""

from __future__ import annotations

import itertools
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

import numpy as np
from numpy.typing import NDArray

from cpa.core.types import (
    SCM,
    Context,
    MCCM,
    AlignmentMapping,
    EdgeClassification,
)
from cpa.core.scm import StructuralCausalModel
from cpa.core.context import ContextSpace


# ===================================================================
# MultiContextCausalModel
# ===================================================================


class MultiContextCausalModel:
    """Full-featured MCCM with variable alignment and batch operations.

    This class wraps the lightweight :class:`MCCM` dataclass with
    richer functionality for cross-context comparison, variable
    alignment, divergence computation, and serialisation.

    Parameters
    ----------
    context_space : ContextSpace, optional
        Context space object.
    mode : ``"union"`` or ``"intersection"``
        How to handle variable mismatches across contexts.

    Examples
    --------
    >>> mccm = MultiContextCausalModel(mode="intersection")
    >>> mccm.add_context(ctx1, scm1)
    >>> mccm.add_context(ctx2, scm2)
    >>> mccm.shared_variables()
    {'X1', 'X2', 'X3'}
    """

    def __init__(
        self,
        context_space: Optional[ContextSpace] = None,
        *,
        mode: str = "intersection",
    ) -> None:
        if mode not in ("union", "intersection"):
            raise ValueError(f"mode must be 'union' or 'intersection', got {mode!r}")
        self._mode = mode
        self._cs = context_space or ContextSpace()
        self._scms: Dict[str, StructuralCausalModel] = {}
        self._metadata: Dict[str, Dict[str, Any]] = {}

    # -----------------------------------------------------------------
    # Properties
    # -----------------------------------------------------------------

    @property
    def num_contexts(self) -> int:
        """Number of contexts in the model."""
        return len(self._scms)

    @property
    def context_ids(self) -> List[str]:
        """List of context identifiers."""
        return list(self._scms.keys())

    @property
    def context_space(self) -> ContextSpace:
        """The underlying context space."""
        return self._cs

    @property
    def mode(self) -> str:
        """Variable alignment mode ('union' or 'intersection')."""
        return self._mode

    # -----------------------------------------------------------------
    # Context management
    # -----------------------------------------------------------------

    def add_context(
        self,
        context: Context,
        scm: StructuralCausalModel,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Add a context with its SCM.

        Parameters
        ----------
        context : Context
            Context descriptor.
        scm : StructuralCausalModel
            Structural causal model for this context.
        metadata : dict, optional
            Additional metadata.

        Raises
        ------
        ValueError
            If context id already exists.
        """
        if context.id in self._scms:
            raise ValueError(f"Context {context.id!r} already exists")
        if context.id not in self._cs:
            self._cs.add(context)
        self._scms[context.id] = scm
        self._metadata[context.id] = metadata or {}

    def remove_context(self, context_id: str) -> StructuralCausalModel:
        """Remove a context and return its SCM.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        StructuralCausalModel

        Raises
        ------
        KeyError
        """
        if context_id not in self._scms:
            raise KeyError(f"Context {context_id!r} not found")
        scm = self._scms.pop(context_id)
        self._metadata.pop(context_id, None)
        if context_id in self._cs:
            self._cs.remove(context_id)
        return scm

    def get_scm(self, context_id: str) -> StructuralCausalModel:
        """Retrieve the SCM for a context.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        StructuralCausalModel

        Raises
        ------
        KeyError
        """
        if context_id not in self._scms:
            raise KeyError(f"Context {context_id!r} not found")
        return self._scms[context_id]

    def get_context(self, context_id: str) -> Context:
        """Retrieve the Context object.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        Context
        """
        return self._cs[context_id]

    # -----------------------------------------------------------------
    # Variable alignment
    # -----------------------------------------------------------------

    def variable_union(self) -> Set[str]:
        """Union of all variable names across contexts.

        Returns
        -------
        set of str
        """
        result: set[str] = set()
        for scm in self._scms.values():
            result |= set(scm.variable_names)
        return result

    def variable_intersection(self) -> Set[str]:
        """Intersection of variable names across contexts.

        Returns
        -------
        set of str
        """
        if not self._scms:
            return set()
        sets = [set(scm.variable_names) for scm in self._scms.values()]
        return sets[0].intersection(*sets[1:])

    def shared_variables(self) -> Set[str]:
        """Variables shared across all contexts (alias for intersection).

        Returns
        -------
        set of str
        """
        return self.variable_intersection()

    def context_specific_variables(
        self, context_id: str
    ) -> Set[str]:
        """Variables unique to *context_id*.

        Parameters
        ----------
        context_id : str

        Returns
        -------
        set of str
        """
        scm = self.get_scm(context_id)
        own = set(scm.variable_names)
        others: set[str] = set()
        for cid, s in self._scms.items():
            if cid != context_id:
                others |= set(s.variable_names)
        return own - others

    def effective_variables(self) -> Set[str]:
        """Variables used for analysis based on the current mode.

        Returns
        -------
        set of str
        """
        if self._mode == "union":
            return self.variable_union()
        return self.variable_intersection()

    def variable_presence_matrix(self) -> Tuple[List[str], NDArray[np.bool_]]:
        """Binary matrix showing which variables are in which contexts.

        Returns
        -------
        (variable_names, presence) : tuple
            variable_names: sorted list of all variables.
            presence: boolean array (num_contexts × num_variables).
        """
        all_vars = sorted(self.variable_union())
        ctx_ids = self.context_ids
        presence = np.zeros((len(ctx_ids), len(all_vars)), dtype=bool)
        for i, cid in enumerate(ctx_ids):
            scm_vars = set(self._scms[cid].variable_names)
            for j, v in enumerate(all_vars):
                presence[i, j] = v in scm_vars
        return all_vars, presence

    # -----------------------------------------------------------------
    # Aligned subgraph extraction
    # -----------------------------------------------------------------

    def aligned_scm_pair(
        self,
        ctx_a: str,
        ctx_b: str,
    ) -> Tuple[StructuralCausalModel, StructuralCausalModel, List[str]]:
        """Extract aligned SCMs restricted to shared variables.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context ids.

        Returns
        -------
        (scm_a, scm_b, shared_names) : tuple
            Subgraph SCMs and the list of shared variable names.
        """
        scm_a = self.get_scm(ctx_a)
        scm_b = self.get_scm(ctx_b)
        shared = sorted(set(scm_a.variable_names) & set(scm_b.variable_names))
        if not shared:
            raise ValueError(
                f"Contexts {ctx_a!r} and {ctx_b!r} share no variables"
            )
        idx_a = [scm_a.variable_index(v) for v in shared]
        idx_b = [scm_b.variable_index(v) for v in shared]
        return scm_a.subgraph(idx_a), scm_b.subgraph(idx_b), shared

    # -----------------------------------------------------------------
    # Pairwise comparison
    # -----------------------------------------------------------------

    def pairwise_shd(self) -> NDArray[np.float64]:
        """Compute pairwise SHD matrix across contexts.

        Only considers shared variables for each pair.

        Returns
        -------
        np.ndarray
            SHD matrix, shape ``(K, K)`` where K = num_contexts.
        """
        K = self.num_contexts
        ids = self.context_ids
        D = np.zeros((K, K), dtype=np.float64)
        for i in range(K):
            for j in range(i + 1, K):
                try:
                    a, b, _ = self.aligned_scm_pair(ids[i], ids[j])
                    shd = a.structural_hamming_distance(b)
                    D[i, j] = shd
                    D[j, i] = shd
                except ValueError:
                    D[i, j] = float("nan")
                    D[j, i] = float("nan")
        return D

    def pairwise_edge_comparison(
        self, ctx_a: str, ctx_b: str
    ) -> Dict[str, Set[Tuple[str, str]]]:
        """Compare edges between two contexts on shared variables.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context ids.

        Returns
        -------
        dict
            Keys: ``"shared"``, ``"modified"``, ``"context_specific_a"``,
            ``"context_specific_b"``.
        """
        scm_a, scm_b, shared = self.aligned_scm_pair(ctx_a, ctx_b)
        edges_a = scm_a.named_edge_set()
        edges_b = scm_b.named_edge_set()

        shared_edges = edges_a & edges_b
        only_a = edges_a - edges_b
        only_b = edges_b - edges_a

        # Check for modified (same edge, different coefficient)
        modified: set[tuple[str, str]] = set()
        truly_shared: set[tuple[str, str]] = set()
        for e in shared_edges:
            src_a, tgt_a = scm_a.variable_index(e[0]), scm_a.variable_index(e[1])
            src_b, tgt_b = scm_b.variable_index(e[0]), scm_b.variable_index(e[1])
            coef_a = scm_a.regression_coefficients[src_a, tgt_a]
            coef_b = scm_b.regression_coefficients[src_b, tgt_b]
            if abs(coef_a - coef_b) > 1e-6:
                modified.add(e)
            else:
                truly_shared.add(e)

        return {
            "shared": truly_shared,
            "modified": modified,
            "context_specific_a": only_a,
            "context_specific_b": only_b,
        }

    def alignment_mapping(
        self, ctx_a: str, ctx_b: str
    ) -> AlignmentMapping:
        """Compute a full alignment mapping between two contexts.

        Parameters
        ----------
        ctx_a, ctx_b : str
            Context ids.

        Returns
        -------
        AlignmentMapping
        """
        scm_a, scm_b, shared = self.aligned_scm_pair(ctx_a, ctx_b)
        edge_comp = self.pairwise_edge_comparison(ctx_a, ctx_b)

        # Mapping: shared variables map to themselves
        pi = {v: v for v in shared}

        # Quality score
        total_edges = sum(len(s) for s in edge_comp.values())
        shared_count = len(edge_comp["shared"])
        quality = shared_count / total_edges if total_edges > 0 else 1.0

        # SHD as structural divergence
        shd = scm_a.structural_hamming_distance(scm_b)

        return AlignmentMapping(
            pi=pi,
            quality_score=quality,
            edge_partition=edge_comp,
            structural_divergence=float(shd),
        )

    # -----------------------------------------------------------------
    # Batch operations
    # -----------------------------------------------------------------

    def apply_to_all(
        self,
        fn: Callable[[str, StructuralCausalModel], Any],
    ) -> Dict[str, Any]:
        """Apply a function to each context's SCM.

        Parameters
        ----------
        fn : callable
            Function with signature ``(context_id, scm) → result``.

        Returns
        -------
        dict
            ``{context_id: result}``.
        """
        return {cid: fn(cid, scm) for cid, scm in self._scms.items()}

    def apply_pairwise(
        self,
        fn: Callable[
            [str, StructuralCausalModel, str, StructuralCausalModel], Any
        ],
    ) -> Dict[Tuple[str, str], Any]:
        """Apply a function to all pairs of contexts.

        Parameters
        ----------
        fn : callable
            Function with signature
            ``(id_a, scm_a, id_b, scm_b) → result``.

        Returns
        -------
        dict
            ``{(id_a, id_b): result}``.
        """
        results: dict[tuple[str, str], Any] = {}
        ids = self.context_ids
        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                results[(ids[i], ids[j])] = fn(
                    ids[i], self._scms[ids[i]],
                    ids[j], self._scms[ids[j]],
                )
        return results

    def filter_contexts(
        self, predicate: Callable[[str, StructuralCausalModel], bool]
    ) -> "MultiContextCausalModel":
        """Create a new MCCM with only contexts passing the predicate.

        Parameters
        ----------
        predicate : callable
            ``(context_id, scm) → bool``.

        Returns
        -------
        MultiContextCausalModel
        """
        new_mccm = MultiContextCausalModel(mode=self._mode)
        for cid, scm in self._scms.items():
            if predicate(cid, scm):
                ctx = self._cs[cid]
                new_mccm.add_context(ctx, scm, metadata=self._metadata.get(cid))
        return new_mccm

    # -----------------------------------------------------------------
    # Summary statistics
    # -----------------------------------------------------------------

    def edge_count_summary(self) -> Dict[str, int]:
        """Edge count per context.

        Returns
        -------
        dict
            ``{context_id: num_edges}``.
        """
        return {cid: scm.num_edges for cid, scm in self._scms.items()}

    def density_summary(self) -> Dict[str, float]:
        """Density per context.

        Returns
        -------
        dict
            ``{context_id: density}``.
        """
        return {cid: scm.density() for cid, scm in self._scms.items()}

    def shared_structure_fraction(self) -> float:
        """Fraction of edges shared across all contexts.

        Returns
        -------
        float
            In [0, 1].
        """
        if self.num_contexts < 2:
            return 1.0
        shared = self.shared_variables()
        if not shared:
            return 0.0

        # Find edges shared across ALL contexts
        shared_list = sorted(shared)
        edge_counts: Dict[Tuple[str, str], int] = defaultdict(int)
        total_unique: set[tuple[str, str]] = set()

        for cid, scm in self._scms.items():
            idx_map = {}
            for v in shared_list:
                if v in scm.variable_names:
                    idx_map[v] = scm.variable_index(v)
            for va in shared_list:
                for vb in shared_list:
                    if va == vb:
                        continue
                    if va in idx_map and vb in idx_map:
                        if scm.has_edge(idx_map[va], idx_map[vb]):
                            edge_counts[(va, vb)] += 1
                            total_unique.add((va, vb))

        if not total_unique:
            return 1.0
        K = self.num_contexts
        fully_shared = sum(1 for cnt in edge_counts.values() if cnt == K)
        return fully_shared / len(total_unique)

    def divergence_matrix(
        self,
        metric: str = "shd",
    ) -> NDArray[np.float64]:
        """Compute pairwise divergence matrix.

        Parameters
        ----------
        metric : ``"shd"``
            Divergence metric.

        Returns
        -------
        np.ndarray
            Divergence matrix, shape ``(K, K)``.
        """
        if metric == "shd":
            return self.pairwise_shd()
        raise ValueError(f"Unknown metric {metric!r}")

    # -----------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------

    def validate(self) -> List[str]:
        """Run consistency checks and return warnings.

        Returns
        -------
        list of str
        """
        warns: list[str] = []

        for cid, scm in self._scms.items():
            if not scm.is_dag():
                warns.append(f"SCM for context {cid!r} contains a cycle")
            if scm.sample_size < 30:
                warns.append(
                    f"Context {cid!r} has small sample size ({scm.sample_size})"
                )
            if scm.sample_size == 0:
                warns.append(
                    f"Context {cid!r} has sample_size=0 (unfitted?)"
                )

        # Variable consistency checks
        shared = self.variable_intersection()
        union = self.variable_union()
        if len(shared) == 0 and self.num_contexts > 1:
            warns.append("No shared variables across contexts")
        if len(shared) < 0.5 * len(union) and self.num_contexts > 1:
            warns.append(
                f"Less than 50% variable overlap "
                f"({len(shared)}/{len(union)} shared)"
            )

        # Sample size consistency
        sizes = [scm.sample_size for scm in self._scms.values()]
        if sizes:
            max_s, min_s = max(sizes), min(sizes)
            if min_s > 0 and max_s / min_s > 10:
                warns.append(
                    f"Large sample size imbalance: "
                    f"min={min_s}, max={max_s} (ratio {max_s/min_s:.1f})"
                )

        return warns

    # -----------------------------------------------------------------
    # Conversion to/from lightweight MCCM
    # -----------------------------------------------------------------

    def to_mccm(self) -> MCCM:
        """Convert to the lightweight :class:`MCCM` dataclass.

        Returns
        -------
        MCCM
        """
        scm_dicts: dict[str, SCM] = {}
        for cid, scm in self._scms.items():
            scm_dicts[cid] = SCM(
                adjacency_matrix=scm.adjacency_matrix,
                regression_coefficients=scm.regression_coefficients,
                residual_variances=scm.residual_variances,
                variable_names=scm.variable_names,
                sample_size=scm.sample_size,
            )
        shared = sorted(self.variable_intersection())
        ctx_specific: dict[str, list[str]] = {}
        for cid in self.context_ids:
            spec = self.context_specific_variables(cid)
            if spec:
                ctx_specific[cid] = sorted(spec)
        return MCCM(
            scms=scm_dicts,
            context_space=list(self._cs),
            shared_variables=shared,
            context_specific_variables=ctx_specific,
        )

    @classmethod
    def from_mccm(cls, mccm: MCCM, *, mode: str = "intersection") -> "MultiContextCausalModel":
        """Create from a lightweight MCCM dataclass.

        Parameters
        ----------
        mccm : MCCM
        mode : str

        Returns
        -------
        MultiContextCausalModel
        """
        cs = ContextSpace(list(mccm.context_space))
        model = cls(context_space=cs, mode=mode)
        for cid, scm_data in mccm.scms.items():
            scm = StructuralCausalModel(
                adjacency_matrix=scm_data.adjacency_matrix,
                variable_names=scm_data.variable_names,
                regression_coefficients=scm_data.regression_coefficients,
                residual_variances=scm_data.residual_variances,
                sample_size=scm_data.sample_size,
            )
            # Don't re-add context if already in cs
            if cid not in model._scms:
                model._scms[cid] = scm
                model._metadata[cid] = {}
        return model

    # -----------------------------------------------------------------
    # Serialisation
    # -----------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns
        -------
        dict
        """
        return {
            "mode": self._mode,
            "context_space": self._cs.to_dict(),
            "scms": {cid: scm.to_dict() for cid, scm in self._scms.items()},
            "metadata": dict(self._metadata),
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MultiContextCausalModel":
        """Deserialize from dictionary.

        Parameters
        ----------
        d : dict

        Returns
        -------
        MultiContextCausalModel
        """
        cs = ContextSpace.from_dict(d["context_space"])
        model = cls(context_space=cs, mode=d.get("mode", "intersection"))
        for cid, scm_d in d["scms"].items():
            scm = StructuralCausalModel.from_dict(scm_d)
            model._scms[cid] = scm
            model._metadata[cid] = d.get("metadata", {}).get(cid, {})
        return model

    # -----------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------

    def summary(self) -> str:
        """Multi-line summary of the MCCM.

        Returns
        -------
        str
        """
        lines = [
            f"MultiContextCausalModel (mode={self._mode})",
            f"  Contexts: {self.num_contexts}",
            f"  Variable union: {len(self.variable_union())}",
            f"  Variable intersection: {len(self.variable_intersection())}",
        ]
        if self.num_contexts > 1:
            lines.append(
                f"  Shared structure: {self.shared_structure_fraction():.1%}"
            )
        for cid in self.context_ids:
            scm = self._scms[cid]
            lines.append(
                f"  [{cid}] p={scm.num_variables}, "
                f"edges={scm.num_edges}, n={scm.sample_size}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MultiContextCausalModel(K={self.num_contexts}, "
            f"mode={self._mode!r}, "
            f"shared={len(self.variable_intersection())})"
        )


# ===================================================================
# Factory functions
# ===================================================================


def build_mccm_from_data(
    datasets: Dict[str, NDArray[np.floating]],
    adjacency_matrices: Dict[str, NDArray[np.floating]],
    *,
    variable_names: Optional[List[str]] = None,
    contexts: Optional[Dict[str, Context]] = None,
    mode: str = "intersection",
) -> MultiContextCausalModel:
    """Build an MCCM by fitting SCMs from data + known DAG structures.

    Parameters
    ----------
    datasets : dict
        ``{context_id: data_matrix}`` where each matrix is ``(n_k, p)``.
    adjacency_matrices : dict
        ``{context_id: adjacency_matrix}`` known DAG structures.
    variable_names : list of str, optional
        Common variable names.
    contexts : dict, optional
        ``{context_id: Context}`` descriptors.
    mode : str
        Variable alignment mode.

    Returns
    -------
    MultiContextCausalModel
    """
    if set(datasets.keys()) != set(adjacency_matrices.keys()):
        raise ValueError("datasets and adjacency_matrices must have same keys")

    mccm = MultiContextCausalModel(mode=mode)
    for cid in datasets:
        data = np.asarray(datasets[cid], dtype=np.float64)
        adj = np.asarray(adjacency_matrices[cid], dtype=np.float64)
        names = variable_names
        if names is None:
            names = [f"X{i}" for i in range(data.shape[1])]

        scm = StructuralCausalModel.fit_from_data(
            data, adj, variable_names=list(names)
        )
        ctx = (contexts or {}).get(cid, Context(id=cid))
        mccm.add_context(ctx, scm)

    return mccm


def build_mccm_from_scms(
    scms: Dict[str, StructuralCausalModel],
    *,
    contexts: Optional[Dict[str, Context]] = None,
    mode: str = "intersection",
) -> MultiContextCausalModel:
    """Build an MCCM from pre-built SCMs.

    Parameters
    ----------
    scms : dict
        ``{context_id: StructuralCausalModel}``.
    contexts : dict, optional
        ``{context_id: Context}``.
    mode : str
        Variable alignment mode.

    Returns
    -------
    MultiContextCausalModel
    """
    mccm = MultiContextCausalModel(mode=mode)
    for cid, scm in scms.items():
        ctx = (contexts or {}).get(cid, Context(id=cid))
        mccm.add_context(ctx, scm)
    return mccm
