"""Composition of transforms and their information-theoretic properties.

This module models the ways individual transforms can be *composed* inside
an ML pipeline – sequential chaining, parallel branching (``ColumnTransformer``
/ ``FeatureUnion``), conditional gating, and iterative loops – and propagates
tight capacity bounds through the resulting composite.

The **Data-Processing Inequality** (DPI) is the fundamental constraint
exploited here: for a Markov chain  X → Y → Z  we have
    I(X; Z) ≤ I(X; Y)
so the information about X available at Z can never exceed what was
available at the intermediate Y.  This gives us a *chain rule* for
sequential compositions that yields upper bounds on end-to-end pipeline
capacity without needing to know the data distribution.
"""

from __future__ import annotations

import enum
import itertools
import logging
import math
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    FrozenSet,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from taintflow.core.types import OpType, Origin, Severity

from .registry import (
    TransformCategory,
    TransformProperty,
    TransformRegistry,
    TransformRegistryEntry,
    TransformSignature,
    default_registry,
)

if TYPE_CHECKING:
    from taintflow.dag.pidag import PIDAG

logger = logging.getLogger(__name__)

__all__ = [
    "CompositionRule",
    "ComposedTransform",
    "TransformComposer",
    "CapacityCompositor",
    "PipelineDecomposer",
]


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class CompositionRule(enum.Enum):
    """How a set of transforms is combined."""

    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    ITERATIVE = "iterative"

    # ------------------------------------------------------------------

    @classmethod
    def from_str(cls, text: str) -> "CompositionRule":
        key = text.strip().lower().replace("-", "_").replace(" ", "_")
        for member in cls:
            if member.value == key or member.name.lower() == key:
                return member
        raise ValueError(f"Unknown CompositionRule: {text!r}")

    @property
    def applies_dpi(self) -> bool:
        """True when the Data-Processing Inequality applies directly."""
        return self in {CompositionRule.SEQUENTIAL, CompositionRule.ITERATIVE}


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ComposedTransform:
    """A transform built by composing one or more named transforms.

    Parameters
    ----------
    steps:
        Ordered list of canonical transform names that appear in the
        :class:`TransformRegistry`.
    composition_rule:
        How the steps are combined.
    total_capacity_bound_fn:
        A human-readable expression for the tight upper bound on the
        end-to-end mutual information, e.g.
        ``"min(C_step1, C_step2)"`` for a two-step sequential pipeline
        under DPI.
    metadata:
        Arbitrary additional annotations (e.g. sklearn class path).
    """

    steps: Tuple[str, ...]
    composition_rule: CompositionRule
    total_capacity_bound_fn: str = "inf"
    metadata: Dict[str, str] = field(default_factory=dict)

    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        return len(self.steps)

    @property
    def is_trivial(self) -> bool:
        """True when there is only a single step (no real composition)."""
        return self.n_steps <= 1

    def step_names_str(self) -> str:
        sep = " → " if self.composition_rule is CompositionRule.SEQUENTIAL else " ‖ "
        return sep.join(self.steps)

    def summary(self) -> str:
        return (
            f"ComposedTransform({self.composition_rule.value}, "
            f"steps={self.n_steps}, bound={self.total_capacity_bound_fn})"
        )


# ---------------------------------------------------------------------------
# TransformComposer
# ---------------------------------------------------------------------------


class TransformComposer:
    """Compose transforms and compute aggregate properties.

    Given a :class:`TransformRegistry` the composer resolves transform
    names, validates compatibility, and produces a
    :class:`ComposedTransform` with a capacity bound expression.
    """

    def __init__(self, registry: Optional[TransformRegistry] = None) -> None:
        self._registry = registry if registry is not None else default_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compose_sequential(
        self,
        step_names: Sequence[str],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ComposedTransform:
        """Build a sequential composition ``step_0 → step_1 → … → step_n``.

        Under the DPI the capacity of the chain is bounded by the
        *minimum* capacity across all steps::

            C_total ≤ min(C_0, C_1, …, C_n)

        Returns a :class:`ComposedTransform` with the bound expression.
        """
        entries = self._resolve_entries(step_names)
        cap_parts = self._collect_capacity_models(entries)
        bound_fn = self._sequential_bound_expression(cap_parts)

        return ComposedTransform(
            steps=tuple(step_names),
            composition_rule=CompositionRule.SEQUENTIAL,
            total_capacity_bound_fn=bound_fn,
            metadata=metadata or {},
        )

    def compose_parallel(
        self,
        step_names: Sequence[str],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ComposedTransform:
        """Build a parallel composition ``step_0 ‖ step_1 ‖ … ‖ step_n``.

        When transforms operate on disjoint subsets of features and
        their outputs are concatenated, the total capacity is bounded
        by the *sum* of individual capacities::

            C_total ≤ Σ C_i

        If any step shares features the bound loosens to ``max(C_i)``
        (worst case).  We return the conservative sum form.
        """
        entries = self._resolve_entries(step_names)
        cap_parts = self._collect_capacity_models(entries)
        bound_fn = self._parallel_bound_expression(cap_parts)

        return ComposedTransform(
            steps=tuple(step_names),
            composition_rule=CompositionRule.PARALLEL,
            total_capacity_bound_fn=bound_fn,
            metadata=metadata or {},
        )

    def compose_conditional(
        self,
        branches: Mapping[str, Sequence[str]],
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ComposedTransform:
        """Model a conditional/switch-case composition.

        The capacity is bounded by the *maximum* capacity across all
        branches (since the adversary controls which branch is taken)::

            C_total ≤ max(C_branch_1, C_branch_2, …)
        """
        all_steps: list[str] = []
        branch_bounds: list[str] = []
        for label, step_names in branches.items():
            entries = self._resolve_entries(step_names)
            caps = self._collect_capacity_models(entries)
            branch_bounds.append(self._sequential_bound_expression(caps))
            all_steps.extend(step_names)

        bound_fn = "max(" + ", ".join(branch_bounds) + ")"
        return ComposedTransform(
            steps=tuple(all_steps),
            composition_rule=CompositionRule.CONDITIONAL,
            total_capacity_bound_fn=bound_fn,
            metadata=metadata or {},
        )

    def compose_iterative(
        self,
        step_names: Sequence[str],
        n_iterations: int = 1,
        *,
        metadata: Optional[Dict[str, str]] = None,
    ) -> ComposedTransform:
        """Model an iterative (loop) composition.

        An iterated pipeline is equivalent to a sequential chain of
        *n_iterations* copies, so the DPI still applies::

            C_total ≤ min(C_step_i)

        Iteration does *not* increase capacity (DPI), but may increase
        practical leakage if the loop body refits on changing data.
        """
        entries = self._resolve_entries(step_names)
        cap_parts = self._collect_capacity_models(entries)
        inner_bound = self._sequential_bound_expression(cap_parts)

        bound_fn = f"min_over_{n_iterations}_iters({inner_bound})"
        return ComposedTransform(
            steps=tuple(step_names) * n_iterations,
            composition_rule=CompositionRule.ITERATIVE,
            total_capacity_bound_fn=bound_fn,
            metadata=metadata or {},
        )

    # ------------------------------------------------------------------
    # Aggregate property queries
    # ------------------------------------------------------------------

    def compute_composed_capacity(
        self,
        composed: ComposedTransform,
    ) -> str:
        """Return the capacity-bound expression for *composed*."""
        return composed.total_capacity_bound_fn

    def composed_data_dependency(
        self,
        composed: ComposedTransform,
    ) -> str:
        """Return the strongest data-dependency across all steps.

        Ordering: ``all-rows`` > ``row-dependent`` > ``row-independent``.
        """
        dep_rank = {"row-independent": 0, "row-dependent": 1, "all-rows": 2}
        worst = 0
        for name in composed.steps:
            entry = self._registry.lookup(name)
            if entry is not None:
                worst = max(worst, dep_rank.get(entry.signature.data_dependency, 2))
        rank_to_dep = {v: k for k, v in dep_rank.items()}
        return rank_to_dep[worst]

    def any_supervised_fit(self, composed: ComposedTransform) -> bool:
        """Return *True* if any step uses the target during fitting."""
        for name in composed.steps:
            entry = self._registry.lookup(name)
            if entry is not None and entry.properties.fit_uses_target:
                return True
        return False

    def verify_data_processing_inequality(
        self,
        composed: ComposedTransform,
    ) -> List[str]:
        """Check that the composed capacity bound is consistent with DPI.

        Returns a list of warning strings; an empty list means the
        composition satisfies DPI.
        """
        warnings: list[str] = []
        if composed.composition_rule not in (
            CompositionRule.SEQUENTIAL,
            CompositionRule.ITERATIVE,
        ):
            return warnings

        entries = [self._registry.lookup(n) for n in composed.steps]
        valid = [e for e in entries if e is not None]
        if not valid:
            warnings.append("No resolvable entries – cannot verify DPI")
            return warnings

        invertible_chain = all(e.properties.is_invertible for e in valid)
        if invertible_chain:
            return warnings

        has_lossy = any(not e.properties.is_invertible for e in valid)
        if has_lossy and composed.total_capacity_bound_fn == "inf":
            warnings.append(
                "Sequential chain contains lossy steps but capacity bound "
                "is 'inf' – expected a tighter bound via DPI"
            )

        for i, e in enumerate(valid):
            if e.properties.is_invertible:
                continue
            if e.properties.min_info_loss_fn == "0":
                warnings.append(
                    f"Step {i} ({e.name}) is non-invertible but "
                    f"min_info_loss_fn='0' – loss bound may be too loose"
                )

        return warnings

    def decompose_pipeline(
        self,
        composed: ComposedTransform,
    ) -> List[Optional[TransformRegistryEntry]]:
        """Resolve every step name back to its registry entry."""
        return [self._registry.lookup(n) for n in composed.steps]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_entries(
        self, names: Sequence[str]
    ) -> List[TransformRegistryEntry]:
        resolved: list[TransformRegistryEntry] = []
        for name in names:
            entry = self._registry.lookup(name)
            if entry is None:
                logger.warning("Transform %r not found in registry", name)
                entry = self._make_unknown_entry(name)
            resolved.append(entry)
        return resolved

    @staticmethod
    def _collect_capacity_models(
        entries: Sequence[TransformRegistryEntry],
    ) -> List[str]:
        return [e.capacity_model for e in entries]

    @staticmethod
    def _sequential_bound_expression(cap_models: Sequence[str]) -> str:
        if not cap_models:
            return "0"
        if len(cap_models) == 1:
            return f"C({cap_models[0]})"
        parts = ", ".join(f"C({m})" for m in cap_models)
        return f"min({parts})"

    @staticmethod
    def _parallel_bound_expression(cap_models: Sequence[str]) -> str:
        if not cap_models:
            return "0"
        if len(cap_models) == 1:
            return f"C({cap_models[0]})"
        parts = " + ".join(f"C({m})" for m in cap_models)
        return parts

    @staticmethod
    def _make_unknown_entry(name: str) -> TransformRegistryEntry:
        sig = TransformSignature(
            name=name,
            category=TransformCategory.CUSTOM,
            data_dependency="all-rows",
        )
        props = TransformProperty(
            is_monotone=False,
            is_invertible=False,
            preserves_independence=False,
            max_info_gain_fn="inf",
            requires_all_data=True,
        )
        return TransformRegistryEntry(
            signature=sig,
            properties=props,
            capacity_model="unknown",
            notes="Auto-generated placeholder for unregistered transform.",
        )


# ---------------------------------------------------------------------------
# CapacityCompositor
# ---------------------------------------------------------------------------


class CapacityCompositor:
    """Arithmetic over capacity bounds for composite pipelines.

    This class works with *numeric* capacity values (in bits) as well
    as *symbolic* expressions, and applies the relevant combination
    rules depending on the composition topology.
    """

    def __init__(self) -> None:
        self._trace: list[str] = []

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def add_sequential_capacities(
        self,
        capacities: Sequence[float],
    ) -> float:
        """Apply DPI to a sequential chain: ``C_total = min(C_i)``.

        Parameters
        ----------
        capacities:
            Per-step capacity upper bounds in bits.

        Returns
        -------
        The tightest (minimum) bound.
        """
        if not capacities:
            return 0.0
        result = min(capacities)
        self._trace.append(
            f"seq({', '.join(f'{c:.4g}' for c in capacities)}) = {result:.4g}"
        )
        return result

    def merge_parallel_capacities(
        self,
        capacities: Sequence[float],
        *,
        disjoint_features: bool = True,
    ) -> float:
        """Combine capacities for parallel branches.

        Parameters
        ----------
        capacities:
            Per-branch capacity bounds.
        disjoint_features:
            If *True* the branches operate on non-overlapping feature
            sets and the total capacity is the *sum*.  Otherwise
            we conservatively return the *maximum*.

        Returns
        -------
        The combined capacity bound.
        """
        if not capacities:
            return 0.0
        if disjoint_features:
            result = math.fsum(capacities)
            label = "par_disjoint"
        else:
            result = max(capacities)
            label = "par_overlap"
        self._trace.append(
            f"{label}({', '.join(f'{c:.4g}' for c in capacities)}) = {result:.4g}"
        )
        return result

    def apply_dpi_bound(
        self,
        upstream_capacity: float,
        step_capacity: float,
    ) -> float:
        """Apply a single DPI step: ``C_out = min(C_upstream, C_step)``.

        This is the incremental version of :meth:`add_sequential_capacities`
        used when walking the DAG edge-by-edge.
        """
        result = min(upstream_capacity, step_capacity)
        self._trace.append(
            f"dpi(up={upstream_capacity:.4g}, step={step_capacity:.4g}) = {result:.4g}"
        )
        return result

    def compute_total_pipeline_bound(
        self,
        sequential_groups: Sequence[Sequence[float]],
        *,
        top_level_rule: CompositionRule = CompositionRule.SEQUENTIAL,
    ) -> float:
        """Compute the end-to-end capacity of a hierarchical pipeline.

        Parameters
        ----------
        sequential_groups:
            A list of *groups*.  Each group is a list of per-step
            capacities that form a sequential chain.  The groups
            themselves are combined according to *top_level_rule*.
        top_level_rule:
            How groups are combined.  ``SEQUENTIAL`` ⇒ min over groups.
            ``PARALLEL`` ⇒ sum (disjoint) or max (overlapping).
            ``CONDITIONAL`` ⇒ max.
            ``ITERATIVE`` ⇒ same as sequential.

        Returns
        -------
        Combined capacity bound in bits.
        """
        group_bounds: list[float] = []
        for group in sequential_groups:
            group_bounds.append(self.add_sequential_capacities(group))

        if top_level_rule in (CompositionRule.SEQUENTIAL, CompositionRule.ITERATIVE):
            total = min(group_bounds) if group_bounds else 0.0
        elif top_level_rule is CompositionRule.PARALLEL:
            total = math.fsum(group_bounds)
        elif top_level_rule is CompositionRule.CONDITIONAL:
            total = max(group_bounds) if group_bounds else 0.0
        else:
            total = max(group_bounds) if group_bounds else 0.0

        self._trace.append(
            f"total({top_level_rule.value}) = {total:.4g}"
        )
        return total

    # ------------------------------------------------------------------
    # Symbolic helpers
    # ------------------------------------------------------------------

    def sequential_bound_symbolic(
        self, expressions: Sequence[str]
    ) -> str:
        """Return a symbolic ``min(…)`` expression for a sequential chain."""
        if not expressions:
            return "0"
        if len(expressions) == 1:
            return expressions[0]
        return "min(" + ", ".join(expressions) + ")"

    def parallel_bound_symbolic(
        self,
        expressions: Sequence[str],
        *,
        disjoint: bool = True,
    ) -> str:
        """Return a symbolic bound for parallel branches."""
        if not expressions:
            return "0"
        if len(expressions) == 1:
            return expressions[0]
        if disjoint:
            return " + ".join(expressions)
        return "max(" + ", ".join(expressions) + ")"

    def conditional_bound_symbolic(
        self, branch_expressions: Sequence[str]
    ) -> str:
        """Return a symbolic ``max(…)`` for conditional branches."""
        if not branch_expressions:
            return "0"
        if len(branch_expressions) == 1:
            return branch_expressions[0]
        return "max(" + ", ".join(branch_expressions) + ")"

    # ------------------------------------------------------------------
    # Trace / debugging
    # ------------------------------------------------------------------

    @property
    def trace(self) -> List[str]:
        """Return the accumulated derivation trace."""
        return list(self._trace)

    def clear_trace(self) -> None:
        self._trace.clear()

    def trace_summary(self) -> str:
        return "\n".join(self._trace)


# ---------------------------------------------------------------------------
# PipelineDecomposer
# ---------------------------------------------------------------------------


_SKLEARN_PIPELINE_CLASS = "sklearn.pipeline.Pipeline"
_SKLEARN_COLUMN_TRANSFORMER_CLASS = "sklearn.compose.ColumnTransformer"
_SKLEARN_FEATURE_UNION_CLASS = "sklearn.pipeline.FeatureUnion"

# Maps sklearn estimator method names to the corresponding OpType
_FIT_TRANSFORM_METHODS: Dict[str, OpType] = {
    "fit": OpType.FIT,
    "transform": OpType.TRANSFORM_SK,
    "fit_transform": OpType.FIT_TRANSFORM,
    "predict": OpType.PREDICT,
    "fit_predict": OpType.FIT_PREDICT,
}


class PipelineDecomposer:
    """Decompose an sklearn-style pipeline description into individual
    transforms, identify fit/transform patterns, and map each step to
    its :class:`TransformRegistryEntry`.

    The decomposer works on a *structural description* of the pipeline
    (a nested list/dict), **not** on a live sklearn object, so that it
    can be used in static-analysis mode without importing sklearn.
    """

    def __init__(self, registry: Optional[TransformRegistry] = None) -> None:
        self._registry = registry if registry is not None else default_registry()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(
        self,
        pipeline_desc: Mapping,
    ) -> List[ComposedTransform]:
        """Decompose a pipeline description.

        Parameters
        ----------
        pipeline_desc:
            A dict with keys:

            * ``"class"`` – fully-qualified sklearn class name.
            * ``"steps"`` – list of ``{"name": …, "class": …}`` dicts.
            * ``"params"`` – (optional) constructor parameters.

        Returns a list of :class:`ComposedTransform` – one per
        top-level branch / sequential chain identified.
        """
        cls_name = pipeline_desc.get("class", "")
        steps_raw = pipeline_desc.get("steps", [])

        if cls_name == _SKLEARN_PIPELINE_CLASS:
            return [self._decompose_sequential(steps_raw, pipeline_desc)]
        if cls_name == _SKLEARN_COLUMN_TRANSFORMER_CLASS:
            return self._decompose_column_transformer(steps_raw, pipeline_desc)
        if cls_name == _SKLEARN_FEATURE_UNION_CLASS:
            return [self._decompose_feature_union(steps_raw, pipeline_desc)]

        # Fallback: treat as a single-step "pipeline"
        return [self._decompose_single(pipeline_desc)]

    def identify_fit_transform_patterns(
        self,
        pipeline_desc: Mapping,
    ) -> List[Dict[str, str]]:
        """Identify whether each step is called as fit-then-transform,
        fit_transform, or transform-only.

        Returns a list of dicts ``{"name": …, "pattern": …}``.
        """
        steps_raw = pipeline_desc.get("steps", [])
        patterns: list[dict[str, str]] = []
        n = len(steps_raw)
        for i, step in enumerate(steps_raw):
            name = step.get("name", f"step_{i}")
            cls_name = step.get("class", "")
            entry = self._registry.lookup_by_sklearn_class(cls_name)

            if entry is not None and not entry.signature.is_fitted:
                patterns.append({"name": name, "pattern": "stateless_transform"})
                continue

            if i < n - 1:
                patterns.append({"name": name, "pattern": "fit_transform"})
            else:
                pattern = step.get("method", "fit_transform")
                patterns.append({"name": name, "pattern": pattern})

        return patterns

    def map_to_registry(
        self,
        pipeline_desc: Mapping,
    ) -> List[Tuple[str, Optional[TransformRegistryEntry]]]:
        """Map each pipeline step to its registry entry (or ``None``).

        Returns a list of ``(step_name, entry_or_None)`` pairs.
        """
        steps_raw = pipeline_desc.get("steps", [])
        result: list[tuple[str, Optional[TransformRegistryEntry]]] = []
        for i, step in enumerate(steps_raw):
            name = step.get("name", f"step_{i}")
            cls_name = step.get("class", "")
            entry = self._registry.lookup_by_sklearn_class(cls_name)
            if entry is None:
                short = cls_name.rsplit(".", 1)[-1] if cls_name else name
                entry = self._registry.lookup(short)
            result.append((name, entry))
        return result

    def severity_for_step(
        self,
        step_desc: Mapping,
    ) -> Severity:
        """Heuristic severity for a single pipeline step.

        * ``CRITICAL`` if the step is supervised (``fit_uses_target``)
          and is not correctly scoped to training data.
        * ``WARNING`` if the step is fitted and uses all rows.
        * ``NEGLIGIBLE`` otherwise.
        """
        cls_name = step_desc.get("class", "")
        entry = self._registry.lookup_by_sklearn_class(cls_name)
        if entry is None:
            return Severity.WARNING

        if entry.properties.fit_uses_target:
            return Severity.CRITICAL
        if entry.signature.is_fitted and entry.properties.requires_all_data:
            return Severity.WARNING
        return Severity.NEGLIGIBLE

    def full_decomposition_report(
        self,
        pipeline_desc: Mapping,
    ) -> Dict:
        """Return a comprehensive decomposition report.

        Keys in the returned dict:
        * ``composed_transforms`` – list of :class:`ComposedTransform`.
        * ``fit_patterns`` – per-step fit/transform patterns.
        * ``registry_map`` – per-step registry resolution.
        * ``severities`` – per-step severity assessments.
        * ``dpi_warnings`` – DPI verification results.
        """
        composed = self.decompose(pipeline_desc)
        fit_patterns = self.identify_fit_transform_patterns(pipeline_desc)
        reg_map = self.map_to_registry(pipeline_desc)

        steps_raw = pipeline_desc.get("steps", [])
        severities = [
            {"name": s.get("name", f"step_{i}"), "severity": self.severity_for_step(s).value}
            for i, s in enumerate(steps_raw)
        ]

        composer = TransformComposer(self._registry)
        dpi_warnings: list[str] = []
        for ct in composed:
            dpi_warnings.extend(composer.verify_data_processing_inequality(ct))

        return {
            "composed_transforms": [c.summary() for c in composed],
            "fit_patterns": fit_patterns,
            "registry_map": [
                {"name": name, "found": entry is not None, "entry": entry.name if entry else None}
                for name, entry in reg_map
            ],
            "severities": severities,
            "dpi_warnings": dpi_warnings,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _decompose_sequential(
        self,
        steps_raw: Sequence[Mapping],
        pipeline_desc: Mapping,
    ) -> ComposedTransform:
        """Decompose a sequential Pipeline into a ComposedTransform."""
        step_names: list[str] = []
        for i, step in enumerate(steps_raw):
            cls_name = step.get("class", "")
            entry = self._registry.lookup_by_sklearn_class(cls_name)
            if entry is not None:
                step_names.append(entry.name)
            else:
                short = cls_name.rsplit(".", 1)[-1] if cls_name else f"unknown_{i}"
                step_names.append(short)

        composer = TransformComposer(self._registry)
        return composer.compose_sequential(
            step_names,
            metadata={"source_class": _SKLEARN_PIPELINE_CLASS},
        )

    def _decompose_column_transformer(
        self,
        steps_raw: Sequence[Mapping],
        pipeline_desc: Mapping,
    ) -> List[ComposedTransform]:
        """Decompose a ColumnTransformer into parallel branches.

        Each branch may itself be a sequential pipeline, yielding a
        list of :class:`ComposedTransform` objects.
        """
        branches: list[ComposedTransform] = []
        composer = TransformComposer(self._registry)

        for i, step in enumerate(steps_raw):
            sub_steps = step.get("steps", [])
            sub_cls = step.get("class", "")
            if sub_steps:
                branch = self._decompose_sequential(sub_steps, step)
            else:
                entry = self._registry.lookup_by_sklearn_class(sub_cls)
                name = entry.name if entry else sub_cls.rsplit(".", 1)[-1]
                branch = composer.compose_sequential(
                    [name],
                    metadata={"branch_index": str(i)},
                )
            branches.append(branch)

        if branches:
            all_steps = []
            for b in branches:
                all_steps.extend(b.steps)
            cap_compositor = CapacityCompositor()
            bound = cap_compositor.parallel_bound_symbolic(
                [b.total_capacity_bound_fn for b in branches]
            )
            overall = ComposedTransform(
                steps=tuple(all_steps),
                composition_rule=CompositionRule.PARALLEL,
                total_capacity_bound_fn=bound,
                metadata={"source_class": _SKLEARN_COLUMN_TRANSFORMER_CLASS},
            )
            branches.append(overall)

        return branches

    def _decompose_feature_union(
        self,
        steps_raw: Sequence[Mapping],
        pipeline_desc: Mapping,
    ) -> ComposedTransform:
        """Decompose a FeatureUnion (parallel composition)."""
        step_names: list[str] = []
        for i, step in enumerate(steps_raw):
            cls_name = step.get("class", "")
            entry = self._registry.lookup_by_sklearn_class(cls_name)
            if entry is not None:
                step_names.append(entry.name)
            else:
                short = cls_name.rsplit(".", 1)[-1] if cls_name else f"unknown_{i}"
                step_names.append(short)

        composer = TransformComposer(self._registry)
        return composer.compose_parallel(
            step_names,
            metadata={"source_class": _SKLEARN_FEATURE_UNION_CLASS},
        )

    def _decompose_single(
        self,
        step_desc: Mapping,
    ) -> ComposedTransform:
        """Wrap a single estimator as a trivial composed transform."""
        cls_name = step_desc.get("class", "")
        entry = self._registry.lookup_by_sklearn_class(cls_name)
        name = entry.name if entry else cls_name.rsplit(".", 1)[-1]

        composer = TransformComposer(self._registry)
        return composer.compose_sequential(
            [name],
            metadata={"source_class": cls_name},
        )
