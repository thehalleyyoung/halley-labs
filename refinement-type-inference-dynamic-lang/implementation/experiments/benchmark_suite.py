"""Benchmark suite: 50 Python functions with known bug labels.

20 null-safety bugs, 20 tensor shape bugs, 10 correct functions.
Each entry: {code, name, has_null_bug, has_shape_bug, category, description}.
"""

NULL_SAFETY_BUGS = [
    {
        "name": "null_dict_get",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "dict.get() returns Optional, dereferenced without check",
        "code": """
def null_dict_get(d, key):
    val = d.get(key)
    return val.strip()
""",
    },
    {
        "name": "null_conditional_escape",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "None escapes one branch",
        "code": """
def null_conditional_escape(x):
    data = None
    if x > 0:
        data = "hello"
    return data.upper()
""",
    },
    {
        "name": "null_overwrite",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Good value overwritten with None",
        "code": """
def null_overwrite(x):
    x = "hello"
    x = None
    return x.upper()
""",
    },
    {
        "name": "null_append_return",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "list.append returns None",
        "code": """
def null_append_return():
    result = [1, 2].append(3)
    return result.count(1)
""",
    },
    {
        "name": "null_wrong_guard",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Guard checks wrong variable",
        "code": """
def null_wrong_guard(x):
    y = None
    if x is not None:
        return y.strip()
    return ""
""",
    },
    {
        "name": "null_loop_init",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Variable set inside loop body may remain None",
        "code": """
def null_loop_init(items):
    result = None
    for item in items:
        if item > 0:
            result = str(item)
    return result.strip()
""",
    },
    {
        "name": "null_chain_deref",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Method chain on None value",
        "code": """
def null_chain_deref():
    result = None
    return result.lower().strip()
""",
    },
    {
        "name": "null_nested_access",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Nested attr access on maybe-None",
        "code": """
def null_nested_access(outer):
    x = None
    if outer:
        x = outer
    return x.attr.method()
""",
    },
    {
        "name": "null_reassign_branch",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Only one branch sets variable",
        "code": """
def null_reassign_branch(cond):
    x = None
    if cond:
        x = "safe"
    return x.strip()
""",
    },
    {
        "name": "null_simple_deref",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Direct None dereference",
        "code": """
def null_simple_deref():
    x = None
    return x.strip()
""",
    },
    {
        "name": "null_sort_return",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "list.sort returns None",
        "code": """
def null_sort_return():
    data = [3, 1, 2].sort()
    return data[0]
""",
    },
    {
        "name": "null_setdefault_misuse",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Misuse of dict.get after pop",
        "code": """
def null_setdefault_misuse(d):
    d.pop("key", None)
    val = d.get("key")
    return val.lower()
""",
    },
    {
        "name": "null_ternary_none",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Ternary can produce None",
        "code": """
def null_ternary_none(cond):
    x = "hello" if cond else None
    return x.upper()
""",
    },
    {
        "name": "null_find_result",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "str.find returns -1, not None, but pattern is Optional-like",
        "code": """
def null_find_result(items, target):
    found = None
    for item in items:
        if item == target:
            found = item
    return found.upper()
""",
    },
    {
        "name": "null_multiple_returns",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Function returns None on some paths, caller dereferences",
        "code": """
def helper(x):
    if x > 0:
        return str(x)
    return None

def null_multiple_returns(x):
    r = helper(x)
    return r.strip()
""",
    },
    {
        "name": "null_exception_handler",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Variable set in try but used after except may be None",
        "code": """
def null_exception_handler():
    result = None
    try:
        result = "ok"
    except Exception:
        pass
    return result.strip()
""",
    },
    {
        "name": "null_attr_on_none",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Attribute access on explicit None",
        "code": """
def null_attr_on_none():
    obj = None
    return obj.value
""",
    },
    {
        "name": "null_subscript_none",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Subscript on None value",
        "code": """
def null_subscript_none():
    data = None
    return data[0]
""",
    },
    {
        "name": "null_iter_none",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Iterating over None",
        "code": """
def null_iter_none():
    items = None
    for x in items:
        print(x)
""",
    },
    {
        "name": "null_call_none",
        "has_null_bug": True,
        "has_shape_bug": False,
        "category": "null_safety",
        "description": "Calling None as function",
        "code": """
def null_call_none():
    fn = None
    return fn()
""",
    },
]

TENSOR_SHAPE_BUGS = [
    {
        "name": "shape_matmul_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "matmul with incompatible inner dims",
        "code": """
import torch
def shape_matmul_mismatch():
    a = torch.randn(3, 4)
    b = torch.randn(5, 6)
    return a @ b
""",
    },
    {
        "name": "shape_matmul_3d",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Batched matmul with wrong inner dim",
        "code": """
import torch
def shape_matmul_3d():
    a = torch.randn(2, 3, 4)
    b = torch.randn(2, 5, 6)
    return torch.bmm(a, b)
""",
    },
    {
        "name": "shape_add_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Element-wise add with incompatible shapes",
        "code": """
import torch
def shape_add_mismatch():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return a + b
""",
    },
    {
        "name": "shape_reshape_invalid",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Reshape to incompatible total elements",
        "code": """
import torch
def shape_reshape_invalid():
    x = torch.randn(3, 4)
    return x.reshape(5, 3)
""",
    },
    {
        "name": "shape_cat_dim_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "torch.cat with mismatched non-cat dim",
        "code": """
import torch
def shape_cat_dim_mismatch():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return torch.cat([a, b], dim=0)
""",
    },
    {
        "name": "shape_linear_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "nn.Linear input features mismatch",
        "code": """
import torch
import torch.nn as nn
def shape_linear_mismatch():
    layer = nn.Linear(10, 5)
    x = torch.randn(32, 8)
    return layer(x)
""",
    },
    {
        "name": "shape_conv_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Conv2d input channels mismatch",
        "code": """
import torch
import torch.nn as nn
def shape_conv_mismatch():
    conv = nn.Conv2d(3, 16, 3)
    x = torch.randn(1, 1, 32, 32)
    return conv(x)
""",
    },
    {
        "name": "shape_view_invalid",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "view with wrong total elements",
        "code": """
import torch
def shape_view_invalid():
    x = torch.randn(2, 3, 4)
    return x.view(5, 5)
""",
    },
    {
        "name": "shape_attention_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Attention Q/K dimension mismatch",
        "code": """
import torch
def shape_attention_mismatch():
    Q = torch.randn(8, 64, 32)
    K = torch.randn(8, 64, 48)
    return torch.bmm(Q, K.transpose(1, 2))
""",
    },
    {
        "name": "shape_broadcast_fail",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Broadcasting fails on incompatible dims",
        "code": """
import torch
def shape_broadcast_fail():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return a * b
""",
    },
    {
        "name": "shape_mm_wrong_dims",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "torch.mm dimension mismatch",
        "code": """
import torch
def shape_mm_wrong_dims():
    a = torch.randn(4, 3)
    b = torch.randn(4, 5)
    return torch.mm(a, b)
""",
    },
    {
        "name": "shape_reshape_chain",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Chain of reshapes leads to invalid shape",
        "code": """
import torch
def shape_reshape_chain():
    x = torch.randn(6, 4)
    y = x.reshape(8, 3)
    return y
""",
    },
    {
        "name": "shape_stack_ndim",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "torch.stack with different shaped tensors",
        "code": """
import torch
def shape_stack_ndim():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return torch.stack([a, b])
""",
    },
    {
        "name": "shape_mv_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Matrix-vector multiply dimension mismatch",
        "code": """
import torch
def shape_mv_mismatch():
    A = torch.randn(3, 4)
    v = torch.randn(5)
    return torch.mv(A, v)
""",
    },
    {
        "name": "shape_residual_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Residual connection with mismatched shapes",
        "code": """
import torch
import torch.nn as nn
def shape_residual_mismatch():
    x = torch.randn(32, 64)
    layer = nn.Linear(64, 128)
    return x + layer(x)
""",
    },
    {
        "name": "shape_transpose_matmul",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Wrong transpose before matmul",
        "code": """
import torch
def shape_transpose_matmul():
    a = torch.randn(3, 4)
    b = torch.randn(3, 5)
    return a @ b
""",
    },
    {
        "name": "shape_concat_wrong_axis",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Concatenation along wrong axis",
        "code": """
import torch
def shape_concat_wrong_axis():
    a = torch.randn(2, 3)
    b = torch.randn(4, 5)
    return torch.cat([a, b], dim=1)
""",
    },
    {
        "name": "shape_flatten_matmul",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Flatten produces wrong dim for linear layer",
        "code": """
import torch
import torch.nn as nn
def shape_flatten_matmul():
    x = torch.randn(1, 3, 8, 8)
    flat = x.view(1, -1)
    layer = nn.Linear(100, 10)
    return layer(flat)
""",
    },
    {
        "name": "shape_squeeze_error",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "Operations after wrong reshape",
        "code": """
import torch
def shape_squeeze_error():
    x = torch.randn(2, 3, 4)
    y = x.reshape(3, 9)
    z = torch.randn(3, 4)
    return y + z
""",
    },
    {
        "name": "shape_einsum_mismatch",
        "has_null_bug": False,
        "has_shape_bug": True,
        "category": "tensor_shapes",
        "description": "matmul after incompatible reshape",
        "code": """
import torch
def shape_einsum_mismatch():
    a = torch.randn(4, 6)
    b = torch.randn(4, 6)
    return a @ b
""",
    },
]

CORRECT_FUNCTIONS = [
    {
        "name": "safe_none_check",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Proper None guard before use",
        "code": """
def safe_none_check(x):
    if x is not None:
        return x.strip()
    return ""
""",
    },
    {
        "name": "safe_early_return",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Early return on None",
        "code": """
def safe_early_return(x):
    if x is None:
        return ""
    return x.strip()
""",
    },
    {
        "name": "safe_isinstance_guard",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "isinstance guard before method call",
        "code": """
def safe_isinstance_guard(x):
    if isinstance(x, str):
        return x.upper()
    return str(x)
""",
    },
    {
        "name": "safe_truthiness",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Truthiness check guards None",
        "code": """
def safe_truthiness(x):
    if x:
        return x.strip()
    return ""
""",
    },
    {
        "name": "safe_default_value",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Default value prevents None",
        "code": """
def safe_default_value(x):
    x = x or "default"
    return x.upper()
""",
    },
    {
        "name": "safe_assert_not_none",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Assert guards against None",
        "code": """
def safe_assert_not_none(x):
    assert x is not None
    return x.strip()
""",
    },
    {
        "name": "safe_matmul",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Valid matmul dimensions",
        "code": """
import torch
def safe_matmul():
    a = torch.randn(3, 4)
    b = torch.randn(4, 5)
    return a @ b
""",
    },
    {
        "name": "safe_reshape",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Valid reshape preserving elements",
        "code": """
import torch
def safe_reshape():
    x = torch.randn(3, 4)
    return x.reshape(6, 2)
""",
    },
    {
        "name": "safe_broadcast",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Valid broadcasting",
        "code": """
import torch
def safe_broadcast():
    a = torch.randn(3, 4)
    b = torch.randn(1, 4)
    return a + b
""",
    },
    {
        "name": "safe_cat_same_dim",
        "has_null_bug": False,
        "has_shape_bug": False,
        "category": "correct",
        "description": "Valid concatenation",
        "code": """
import torch
def safe_cat_same_dim():
    a = torch.randn(3, 4)
    b = torch.randn(5, 4)
    return torch.cat([a, b], dim=0)
""",
    },
]

ALL_BENCHMARKS = NULL_SAFETY_BUGS + TENSOR_SHAPE_BUGS + CORRECT_FUNCTIONS


def get_benchmarks():
    """Return all 50 benchmark entries."""
    return ALL_BENCHMARKS


def get_by_category(category: str):
    """Return benchmarks filtered by category."""
    return [b for b in ALL_BENCHMARKS if b["category"] == category]
