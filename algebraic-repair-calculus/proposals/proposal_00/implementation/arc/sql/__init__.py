"""
arc.sql — SQL Semantic Analysis
=================================

Provides SQL parsing, column-level lineage analysis, operator models,
Fragment F testing, and predicate analysis for the ARC framework.

Modules:
- parser:     SQL Parser (sqlglot-based)
- lineage:    Column-Level Lineage Analysis
- operators:  SQL Operator Models
- fragment:   Fragment F Testing (Algorithm A6)
- predicates: Predicate Analysis with three-valued logic
"""

from arc.sql.parser import (
    Dialect,
    ParsedQuery,
    SQLParser,
)

from arc.sql.lineage import (
    ColumnLineage,
    ColumnLineageEntry,
    LineageAnalyzer,
    LineageEdge,
    LineageGraph,
    SourceColumn,
    TransformationStep,
    build_lineage_graph,
    trace_impact,
)

from arc.sql.operators import (
    AggregateFunctionType,
    AggregateExprSpec,
    AlgebraicProperties,
    CTEOperator,
    CTESpec,
    ColumnReference,
    ExpressionRef,
    FilterOperator,
    GroupByOperator,
    JoinConditionSpec,
    JoinKind,
    JoinOperator,
    OrderBySpec,
    SQLOperator,
    SQLOperatorType,
    SchemaDerivation,
    SchemaRequirement,
    SelectOperator,
    SetOpOperator,
    SetOperationType,
    TableReference,
    TransformationType,
    UnionOperator,
    WindowFrameKind,
    WindowFrameSpec,
    WindowFunctionSpec,
    WindowOperator,
    check_schema_compatibility,
    create_operator,
    is_operator_deterministic,
    operator_preserves_multiplicities,
)

from arc.sql.fragment import (
    FragmentChecker,
    FragmentResult,
    FragmentViolation,
    NodeFragmentResult,
    ViolationCategory,
    check_fragment_f,
    check_pipeline_fragment_f,
    fragment_f_violations,
    is_deterministic_query,
)

from arc.sql.predicates import (
    AndPredicate,
    ComparisonOp,
    ComparisonPredicate,
    ExistsPredicate,
    FalsePredicate,
    InSubqueryPredicate,
    LiteralValue,
    NotPredicate,
    OrPredicate,
    PredicateAnalyzer,
    PredicateNode,
    ThreeValuedBool,
    TruePredicate,
)

from arc.sql.rewriter import (
    RewriteDialect,
    RewriteResult,
    SchemaDeltaSpec,
    SQLRewriter,
    add_quality_filters,
    generate_diff,
    generate_merge,
    rewrite_for_schema_delta,
)
# Note: QualityConstraint from rewriter conflicts with the name in other modules;
# import under a qualified name if needed.

from arc.sql.catalog import (
    SchemaCatalog,
    SchemaDiff,
    SchemaVersion,
    SchemaViolation as CatalogSchemaViolation,
    ViolationSeverity,
    diff_schemas as catalog_diff_schemas,
    merge_schemas,
    schemas_compatible,
)

__all__ = [
    # Parser
    "Dialect",
    "ParsedQuery",
    "SQLParser",
    # Lineage
    "ColumnLineage",
    "ColumnLineageEntry",
    "LineageAnalyzer",
    "LineageEdge",
    "LineageGraph",
    "SourceColumn",
    "TransformationStep",
    "build_lineage_graph",
    "trace_impact",
    # Operators
    "AggregateFunctionType",
    "AggregateExprSpec",
    "AlgebraicProperties",
    "CTEOperator",
    "CTESpec",
    "ColumnReference",
    "ExpressionRef",
    "FilterOperator",
    "GroupByOperator",
    "JoinConditionSpec",
    "JoinKind",
    "JoinOperator",
    "OrderBySpec",
    "SQLOperator",
    "SQLOperatorType",
    "SchemaDerivation",
    "SchemaRequirement",
    "SelectOperator",
    "SetOpOperator",
    "SetOperationType",
    "TableReference",
    "TransformationType",
    "UnionOperator",
    "WindowFrameKind",
    "WindowFrameSpec",
    "WindowFunctionSpec",
    "WindowOperator",
    "check_schema_compatibility",
    "create_operator",
    "is_operator_deterministic",
    "operator_preserves_multiplicities",
    # Fragment
    "FragmentChecker",
    "FragmentResult",
    "FragmentViolation",
    "NodeFragmentResult",
    "ViolationCategory",
    "check_fragment_f",
    "check_pipeline_fragment_f",
    "fragment_f_violations",
    "is_deterministic_query",
    # Predicates
    "AndPredicate",
    "ComparisonOp",
    "ComparisonPredicate",
    "ExistsPredicate",
    "FalsePredicate",
    "InSubqueryPredicate",
    "LiteralValue",
    "NotPredicate",
    "OrPredicate",
    "PredicateAnalyzer",
    "PredicateNode",
    "ThreeValuedBool",
    "TruePredicate",
    # Rewriter
    "RewriteDialect",
    "RewriteResult",
    "SchemaDeltaSpec",
    "SQLRewriter",
    "add_quality_filters",
    "generate_diff",
    "generate_merge",
    "rewrite_for_schema_delta",
    # Catalog
    "SchemaCatalog",
    "SchemaDiff",
    "SchemaVersion",
    "CatalogSchemaViolation",
    "ViolationSeverity",
    "catalog_diff_schemas",
    "merge_schemas",
    "schemas_compatible",
]
