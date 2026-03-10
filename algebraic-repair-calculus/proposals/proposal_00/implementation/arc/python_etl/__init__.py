"""
``arc.python_etl`` — Python ETL code analysis for lineage extraction.

Provides:

* :class:`PythonETLAnalyzer` – top-level analyzer that detects the ETL
  framework and dispatches to specialised analyzers.
* :class:`PandasAnalyzer` – pandas-specific lineage extraction.
* :class:`PySparkAnalyzer` – PySpark-specific lineage extraction.
* AST utility functions for safe parsing and variable tracking.
"""

from arc.python_etl.analyzer import PythonETLAnalyzer
from arc.python_etl.pandas_analyzer import PandasAnalyzer
from arc.python_etl.pyspark_analyzer import PySparkAnalyzer
from arc.python_etl.dbt_analyzer import (
    DBTAnalyzer,
    DBTModel,
    DBTProject,
    DBTSource,
    DBTTest,
    IncrementalModel,
    MaterializationType,
    TestType,
    analyze_dbt_project,
    build_dbt_lineage,
    extract_dbt_tests,
)

__all__ = [
    "PythonETLAnalyzer",
    "PandasAnalyzer",
    "PySparkAnalyzer",
    # dbt
    "DBTAnalyzer",
    "DBTModel",
    "DBTProject",
    "DBTSource",
    "DBTTest",
    "IncrementalModel",
    "MaterializationType",
    "TestType",
    "analyze_dbt_project",
    "build_dbt_lineage",
    "extract_dbt_tests",
]
