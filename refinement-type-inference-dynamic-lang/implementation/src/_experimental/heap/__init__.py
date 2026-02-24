"""
Python-native heap model for refinement type inference.

This package models Python's heap-based computation model:
- Heap addresses and abstract heaps (heap_model)
- Alias analysis and points-to graphs (alias_analysis)
- Python class system with MRO and descriptors (class_model)
- Mutation tracking and refinement invalidation (mutation_tracking)
- Class hierarchy analysis with C3 linearization (class_hierarchy)
- Descriptor protocol modeling (descriptor_protocol)
- Dunder method modeling (dunder_model)
- Mutation semantics for heap-aware refinement (mutation_semantics)
- Effect system for side-effect tracking (effect_system)
"""

from src.heap.heap_model import (
    HeapAddress,
    HeapObject,
    AbstractHeap,
    AbstractValue,
    RecencyFlag,
)
from src.heap.alias_analysis import AliasSet, PointsToGraph
from src.heap.class_model import PythonClass, DescriptorKind, MROComputer, AttributeResolver
from src.heap.mutation_tracking import MutationTracker, MutationKind
from src.heap.class_hierarchy import (
    ClassHierarchyAnalyzer,
    ClassInfo,
    MethodInfo,
    PropertyInfo,
    ParamInfo,
)
from src.heap.descriptor_protocol import (
    DescriptorAnalyzer,
    DescriptorInfo,
    AttributeResolution,
    DescriptorKind as DescriptorKindNew,
    PropertyType,
    ClassMethodType,
    StaticMethodType,
    SlotInfo,
)
from src.heap.dunder_model import (
    DunderModel,
    BinaryOpResult,
    ComparisonResult,
    ContainerAccessResult,
    ContextManagerType,
    IterationType,
    TruthinessInfo,
)
from src.heap.mutation_semantics import (
    MutationSemantics,
    ModificationFrame,
    MutationEffect,
    RefinementInvalidation,
    ImmutabilityInfo,
    CollectionMutationResult,
)
from src.heap.effect_system import (
    Effect,
    EffectSet,
    EffectAnalyzer,
    FunctionEffectSummary,
    ExpressionEffect,
)

__all__ = [
    "HeapAddress",
    "HeapObject",
    "AbstractHeap",
    "AbstractValue",
    "RecencyFlag",
    "AliasSet",
    "PointsToGraph",
    "PythonClass",
    "DescriptorKind",
    "MROComputer",
    "AttributeResolver",
    "MutationTracker",
    "MutationKind",
    # class_hierarchy
    "ClassHierarchyAnalyzer",
    "ClassInfo",
    "MethodInfo",
    "PropertyInfo",
    "ParamInfo",
    # descriptor_protocol
    "DescriptorAnalyzer",
    "DescriptorInfo",
    "AttributeResolution",
    "DescriptorKindNew",
    "PropertyType",
    "ClassMethodType",
    "StaticMethodType",
    "SlotInfo",
    # dunder_model
    "DunderModel",
    "BinaryOpResult",
    "ComparisonResult",
    "ContainerAccessResult",
    "ContextManagerType",
    "IterationType",
    "TruthinessInfo",
    # mutation_semantics
    "MutationSemantics",
    "ModificationFrame",
    "MutationEffect",
    "RefinementInvalidation",
    "ImmutabilityInfo",
    "CollectionMutationResult",
    # effect_system
    "Effect",
    "EffectSet",
    "EffectAnalyzer",
    "FunctionEffectSummary",
    "ExpressionEffect",
]
