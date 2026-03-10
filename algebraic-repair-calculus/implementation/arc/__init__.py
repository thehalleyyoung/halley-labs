"""
Algebraic Repair Calculus (ARC)
===============================

Three-sorted delta algebra for provably correct data pipeline repair.
Provides algebraic framework Δ = (Δ_S, Δ_D, Δ_Q, ∘, ⁻¹, push) for
reasoning about and repairing data pipeline perturbations.

Modules
-------
arc.types
    Core type system: SQL types, schemas, operators, tuples, multisets.
arc.graph
    Pipeline DAG: nodes, edges, builder, analysis, visualization.
arc.io
    Serialization: JSON/YAML pipeline specs, delta and repair plan I/O.
arc.cli
    Command-line interface (Click-based).
arc.algebra
    Three-sorted delta algebra engine.
arc.sql
    SQL semantic analysis (sqlglot-based).
arc.planner
    Cost-optimal repair planner.
arc.execution
    Saga-based repair executor.
arc.quality
    Data quality monitoring and drift detection.
arc.python_etl
    Python idiom matching for pandas/PySpark.
"""

__version__ = "0.1.0"

from arc.types import *  # noqa: F401,F403
from arc.algebra import *  # noqa: F401,F403
from arc.sql import *  # noqa: F401,F403
from arc.graph import *  # noqa: F401,F403
from arc.planner import *  # noqa: F401,F403
from arc.execution import *  # noqa: F401,F403
from arc.quality import *  # noqa: F401,F403
from arc.python_etl import *  # noqa: F401,F403
from arc.io import *  # noqa: F401,F403
