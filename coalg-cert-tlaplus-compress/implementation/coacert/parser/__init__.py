"""CoaCert TLA-lite parser package.

Re-exports the public API of every sub-module so callers can write::

    from coacert.parser import parse, pretty_print, check_types, Lexer, ...
"""

# Source-location tracking
from .source_map import SourceLocation, SourceMap, SourceRange, UNKNOWN_LOCATION

# Tokens
from .tokens import Token, TokenKind, KEYWORDS, BINARY_OP_TOKENS, UNARY_PREFIX_TOKENS

# Lexer
from .lexer import Lexer, LexerError, tokenize, token_stream

# AST nodes
from .ast_nodes import (
    ASTNode,
    ASTVisitor,
    # Types
    TypeAnnotation,
    IntType,
    BoolType,
    StringType,
    SetType,
    FunctionType,
    TupleType,
    RecordType,
    SequenceType,
    AnyType,
    OperatorType,
    # Operators
    Operator,
    # Expressions
    Expression,
    IntLiteral,
    BoolLiteral,
    StringLiteral,
    Identifier,
    PrimedIdentifier,
    OperatorApplication,
    SetEnumeration,
    SetComprehension,
    FunctionConstruction,
    FunctionApplication,
    RecordConstruction,
    RecordAccess,
    TupleLiteral,
    SequenceLiteral,
    QuantifiedExpr,
    IfThenElse,
    LetIn,
    CaseArm,
    CaseExpr,
    UnchangedExpr,
    ExceptExpr,
    ChooseExpr,
    DomainExpr,
    # Definitions
    Definition,
    OperatorDef,
    FunctionDef,
    VariableDecl,
    ConstantDecl,
    Assumption,
    Theorem,
    InstanceDef,
    # Action
    ActionExpr,
    StutteringAction,
    FairnessExpr,
    # Temporal
    AlwaysExpr,
    EventuallyExpr,
    LeadsToExpr,
    TemporalForallExpr,
    TemporalExistsExpr,
    # Properties
    Property,
    InvariantProperty,
    TemporalProperty,
    SafetyProperty,
    LivenessProperty,
    # Module
    Module,
)

# Parser
from .parser import Parser, ParseError, parse, parse_expression

# Pretty-printer
from .pretty_printer import PrettyPrinter, pretty_print

# Type checker
from .type_checker import TypeChecker, TypeEnv, TypeError_, check_types

__all__ = [
    # source_map
    "SourceLocation", "SourceMap", "SourceRange", "UNKNOWN_LOCATION",
    # tokens
    "Token", "TokenKind", "KEYWORDS", "BINARY_OP_TOKENS", "UNARY_PREFIX_TOKENS",
    # lexer
    "Lexer", "LexerError", "tokenize", "token_stream",
    # ast_nodes
    "ASTNode", "ASTVisitor", "Operator",
    "TypeAnnotation", "IntType", "BoolType", "StringType", "SetType",
    "FunctionType", "TupleType", "RecordType", "SequenceType", "AnyType",
    "OperatorType",
    "Expression", "IntLiteral", "BoolLiteral", "StringLiteral",
    "Identifier", "PrimedIdentifier", "OperatorApplication",
    "SetEnumeration", "SetComprehension",
    "FunctionConstruction", "FunctionApplication",
    "RecordConstruction", "RecordAccess",
    "TupleLiteral", "SequenceLiteral",
    "QuantifiedExpr", "IfThenElse", "LetIn", "CaseArm", "CaseExpr",
    "UnchangedExpr", "ExceptExpr", "ChooseExpr", "DomainExpr",
    "Definition", "OperatorDef", "FunctionDef",
    "VariableDecl", "ConstantDecl", "Assumption", "Theorem", "InstanceDef",
    "ActionExpr", "StutteringAction", "FairnessExpr",
    "AlwaysExpr", "EventuallyExpr", "LeadsToExpr",
    "TemporalForallExpr", "TemporalExistsExpr",
    "Property", "InvariantProperty", "TemporalProperty",
    "SafetyProperty", "LivenessProperty",
    "Module",
    # parser
    "Parser", "ParseError", "parse", "parse_expression",
    # pretty_printer
    "PrettyPrinter", "pretty_print",
    # type_checker
    "TypeChecker", "TypeEnv", "TypeError_", "check_types",
]
