"""
Standard library models for refinement type inference.

Provides refinement-typed signatures for Python's standard library,
enabling the inference engine to reason about built-in functions,
common modules (os, sys, json, re, math, etc.), and their
pre/postconditions, side effects, and exception behavior.

Usage::

    from src.stdlib import ModelRegistry, get_global_registry
    registry = get_global_registry()
    model = registry.lookup("builtins.len")
    sig = model.get_signature()
"""

from __future__ import annotations

from src.stdlib.python_models import (
    # Core base classes
    StdlibModel,
    ModelRegistry,
    # Signature / constraint descriptors
    ParamSpec,
    ReturnSpec,
    SideEffect,
    ExceptionSpec,
    RefinementConstraint,
    FunctionSignature,
    # Registry access
    get_global_registry,
    register_all_models,
    # Module-level registrars
    register_builtin_models,
    register_os_models,
    register_sys_models,
    register_json_models,
    register_re_models,
    register_math_models,
    register_collections_models,
    register_itertools_models,
    register_functools_models,
    register_typing_models,
    register_pathlib_models,
)

__all__ = [
    "StdlibModel",
    "ModelRegistry",
    "ParamSpec",
    "ReturnSpec",
    "SideEffect",
    "ExceptionSpec",
    "RefinementConstraint",
    "FunctionSignature",
    "get_global_registry",
    "register_all_models",
]
