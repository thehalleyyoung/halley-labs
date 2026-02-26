"""
TLA-lite semantic engine for CoaCert-TLA.

This package provides a complete runtime for evaluating TLA+ expressions,
computing next-state relations, and exploring state spaces.

Submodules
----------
values
    Immutable TLA+ runtime value types (Int, Bool, Set, Function, …).
state
    State representation with hashing, fingerprinting, and state-space tracking.
environment
    Scoped evaluation environment with constant/operator resolution.
evaluator
    Expression evaluator dispatching over the AST.
actions
    Action evaluator computing successor states from TLA+ actions.
type_system
    Runtime type inference and checking for TLA+ values.
builtins
    Built-in operator implementations for standard TLA+ modules.
"""

# --- values ---------------------------------------------------------------
from .values import (
    TLAValue,
    TLAValueError,
    IntValue,
    BoolValue,
    StringValue,
    SetValue,
    FunctionValue,
    TupleValue,
    RecordValue,
    SequenceValue,
    ModelValue,
    value_from_python,
    values_to_json_string,
    value_from_json_string,
)

# --- state ----------------------------------------------------------------
from .state import (
    TLAState,
    StateSignature,
    StateSpace,
)

# --- environment ----------------------------------------------------------
from .environment import (
    Environment,
    OpDef,
    ConstantDecl,
    BuiltinEntry,
)

# --- evaluator ------------------------------------------------------------
from .evaluator import (
    Expr,
    ExprKind,
    EvalError,
    evaluate,
    # AST constructors
    int_lit,
    bool_lit,
    string_lit,
    name_ref,
    primed_ref,
    unary_op,
    binary_op,
    if_then_else,
    let_in,
    case_expr,
    quant_forall,
    quant_exists,
    choose,
    set_enum,
    set_comp,
    set_filter,
    func_construct,
    func_apply,
    func_except,
    domain_op,
    tuple_construct,
    record_construct,
    record_access,
    record_except,
    seq_op,
    op_apply,
    builtin_call,
    unchanged,
)

# --- actions --------------------------------------------------------------
from .actions import (
    ActionExpr,
    ActionEvaluator,
    action_conj,
    action_disj,
    action_exists,
    action_unchanged,
    action_from_expr,
    action_seq,
    compute_successors,
    compute_initial_states,
    explore_state_space,
    is_action_enabled,
    detect_stuttering,
)

# --- type system ----------------------------------------------------------
from .type_system import (
    TLAType,
    TypeKind,
    TypeChecker,
    ANY_TYPE,
    BOOL_TYPE,
    INT_TYPE,
    STRING_TYPE,
    NONE_TYPE,
    set_type,
    function_type,
    tuple_type,
    record_type,
    sequence_type,
    union_type,
    model_type,
)

# --- builtins -------------------------------------------------------------
from .builtins import (
    ModuleRegistry,
    get_default_registry,
    install_standard_modules,
    install_module,
)

__all__ = [
    # values
    "TLAValue", "TLAValueError",
    "IntValue", "BoolValue", "StringValue",
    "SetValue", "FunctionValue", "TupleValue",
    "RecordValue", "SequenceValue", "ModelValue",
    "value_from_python", "values_to_json_string", "value_from_json_string",
    # state
    "TLAState", "StateSignature", "StateSpace",
    # environment
    "Environment", "OpDef", "ConstantDecl", "BuiltinEntry",
    # evaluator
    "Expr", "ExprKind", "EvalError", "evaluate",
    "int_lit", "bool_lit", "string_lit", "name_ref", "primed_ref",
    "unary_op", "binary_op", "if_then_else", "let_in", "case_expr",
    "quant_forall", "quant_exists", "choose",
    "set_enum", "set_comp", "set_filter",
    "func_construct", "func_apply", "func_except", "domain_op",
    "tuple_construct", "record_construct", "record_access", "record_except",
    "seq_op", "op_apply", "builtin_call", "unchanged",
    # actions
    "ActionExpr", "ActionEvaluator",
    "action_conj", "action_disj", "action_exists",
    "action_unchanged", "action_from_expr", "action_seq",
    "compute_successors", "compute_initial_states",
    "explore_state_space", "is_action_enabled", "detect_stuttering",
    # type system
    "TLAType", "TypeKind", "TypeChecker",
    "ANY_TYPE", "BOOL_TYPE", "INT_TYPE", "STRING_TYPE", "NONE_TYPE",
    "set_type", "function_type", "tuple_type",
    "record_type", "sequence_type", "union_type", "model_type",
    # builtins
    "ModuleRegistry", "get_default_registry",
    "install_standard_modules", "install_module",
]
