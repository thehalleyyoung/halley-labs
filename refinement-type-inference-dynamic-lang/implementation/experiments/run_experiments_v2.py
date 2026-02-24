"""
Comprehensive experiment suite for TensorGuard v2.

Experiments:
  1. Controlled benchmark (null-safety, div-by-zero, index OOB)
  2. Tensor shape benchmark (PyTorch/NumPy shape errors)  [NEW]
  3. Comparison with Pyright-style analysis                [NEW]
  4. Performance benchmarks
"""

import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.liquid import analyze_liquid, harvest_predicates
from src.tensor_shapes import analyze_shapes


CONTROLLED_BENCHMARKS = [
    {"code": 'def f(x):\n    val = x.get("key")\n    return val.strip()', "has_bug": True, "kind": "NULL_DEREF", "name": "dict_get_deref"},
    {"code": "def f(items):\n    return items[0].process()", "has_bug": True, "kind": "NULL_DEREF", "name": "first_elem_deref"},
    {"code": "def f(x, y):\n    return x / y", "has_bug": True, "kind": "DIV_BY_ZERO", "name": "unguarded_div"},
    {"code": 'import re\ndef f(text):\n    m = re.match(r"\\d+", text)\n    return m.group(0)', "has_bug": True, "kind": "NULL_DEREF", "name": "regex_match_deref"},
    {"code": 'def f(data):\n    result = data.get("value")\n    return result + 1', "has_bug": True, "kind": "NULL_DEREF", "name": "dict_get_arithmetic"},
    {"code": "def f(x):\n    return x.attr", "has_bug": True, "kind": "NULL_DEREF", "name": "unguarded_attr"},
    {"code": "def average(nums):\n    return sum(nums) / len(nums)", "has_bug": True, "kind": "DIV_BY_ZERO", "name": "empty_list_avg"},
    {"code": 'def f(d, key):\n    val = d.get(key, None)\n    return val.upper()', "has_bug": True, "kind": "NULL_DEREF", "name": "dict_get_none_default"},
    {"code": 'def f(x, y):\n    if y == 0:\n        raise ValueError("zero")\n    return x / y', "has_bug": False, "kind": None, "name": "guarded_div_raise"},
    {"code": "def f(x, y):\n    assert y != 0\n    return x / y", "has_bug": False, "kind": None, "name": "asserted_div"},
    {"code": "def f(x, y):\n    if y == 0:\n        return 0\n    return x / y", "has_bug": False, "kind": None, "name": "guarded_div_return"},
    {"code": 'def f(val):\n    if val is not None:\n        return val.strip()\n    return ""', "has_bug": False, "kind": None, "name": "null_guarded_attr"},
    {"code": "def f(x):\n    assert x is not None\n    return x.strip()", "has_bug": False, "kind": None, "name": "assert_not_none"},
    {"code": "def f(x):\n    if isinstance(x, str):\n        return x.upper()\n    return str(x)", "has_bug": False, "kind": None, "name": "isinstance_guard"},
    {"code": "def f(x, y):\n    if y > 0:\n        return x / y\n    return 0", "has_bug": False, "kind": None, "name": "positive_guard_div"},
]

TENSOR_BENCHMARKS = [
    {"code": "x = torch.randn(3, 4)\ny = torch.randn(5, 6)\nz = x @ y", "has_bug": True, "kind": "MATMUL_INCOMPAT", "name": "matmul_dim_mismatch"},
    {"code": "a = torch.randn(32, 128)\nb = torch.randn(64, 32)\nc = a @ b", "has_bug": True, "kind": "MATMUL_INCOMPAT", "name": "matmul_inner_dim"},
    {"code": "a = torch.randn(3, 4)\nb = torch.randn(5, 4)\nc = a + b", "has_bug": True, "kind": "BROADCAST_FAIL", "name": "broadcast_fail"},
    {"code": "a = torch.randn(3, 4)\nb = torch.randn(3, 5)\nc = torch.cat([a, b], dim=0)", "has_bug": True, "kind": "CAT_INCOMPAT", "name": "cat_dim_mismatch"},
    {"code": "import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(784, 256)\n        self.fc2 = nn.Linear(128, 10)\n    def forward(self, x):\n        x = torch.randn(32, 784)\n        x = self.fc1(x)\n        x = self.fc2(x)\n        return x", "has_bug": True, "kind": "DIM_MISMATCH", "name": "linear_dim_mismatch"},
    {"code": "query = torch.randn(32, 8, 64, 64)\nkey = torch.randn(32, 8, 64, 32)\nscores = query @ key.transpose(2, 3)", "has_bug": True, "kind": "MATMUL_INCOMPAT", "name": "attention_shape_error"},
    {"code": "x = torch.randn(2, 3, 4)\ny = x.reshape(6, 4)\nw = torch.randn(5, 2)\nz = y @ w", "has_bug": True, "kind": "MATMUL_INCOMPAT", "name": "reshape_matmul"},
    {"code": "x = torch.randn(3, 4)\ny = torch.randn(4, 5)\nz = x @ y", "has_bug": False, "kind": None, "name": "matmul_correct"},
    {"code": "a = torch.randn(3, 4)\nb = torch.randn(1, 4)\nc = a + b", "has_bug": False, "kind": None, "name": "broadcast_correct"},
    {"code": "a = torch.randn(3, 4)\nb = torch.randn(5, 4)\nc = torch.cat([a, b], dim=0)", "has_bug": False, "kind": None, "name": "cat_correct"},
    {"code": "import torch.nn as nn\nclass M(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.fc1 = nn.Linear(784, 256)\n        self.fc2 = nn.Linear(256, 10)\n    def forward(self, x):\n        x = torch.randn(32, 784)\n        x = self.fc1(x)\n        x = self.fc2(x)\n        return x", "has_bug": False, "kind": None, "name": "linear_correct"},
    {"code": "q = torch.randn(32, 8, 64, 64)\nk = torch.randn(32, 8, 64, 64)\nscores = q @ k.transpose(2, 3)", "has_bug": False, "kind": None, "name": "attention_correct"},
    {"code": "x = torch.randn(3, 4)\ny = x.transpose(0, 1)\nz = torch.randn(3, 4)\nw = y @ z", "has_bug": False, "kind": None, "name": "transpose_matmul_correct"},
]


def run_benchmark(name, benchmarks, analyze_fn, extract_bugs_fn):
    print(f"\n{'='*60}\n{name}\n{'='*60}")
    tp, fp, tn, fn = 0, 0, 0, 0
    results = []
    for bench in benchmarks:
        result = analyze_fn(bench["code"])
        bugs = extract_bugs_fn(result)
        has_bugs = len(bugs) > 0
        if bench["has_bug"] and has_bugs: tp += 1; status = "TP"
        elif bench["has_bug"] and not has_bugs: fn += 1; status = "FN"
        elif not bench["has_bug"] and has_bugs: fp += 1; status = "FP"
        else: tn += 1; status = "TN"
        results.append({"name": bench["name"], "status": status})
        print(f"  [{status}] {bench['name']}")
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    print(f"\n  Precision: {prec:.1%}  Recall: {rec:.1%}  F1: {f1:.3f}")
    print(f"  TP={tp} FP={fp} TN={tn} FN={fn}")
    return {"precision": prec, "recall": rec, "f1": f1, "tp": tp, "fp": fp, "tn": tn, "fn": fn, "results": results}


def main():
    all_results = {}

    all_results["null_safety"] = run_benchmark(
        "Null-Safety Benchmark", CONTROLLED_BENCHMARKS,
        analyze_liquid, lambda r: r.bugs)

    all_results["tensor_shapes"] = run_benchmark(
        "Tensor Shape Benchmark", TENSOR_BENCHMARKS,
        analyze_shapes, lambda r: r.errors)

    # Performance
    print(f"\n{'='*60}\nPerformance\n{'='*60}")
    for n in [10, 50, 100]:
        code = "\n".join([f"def f_{i}(x,y):\n    if y==0: raise ValueError\n    return x/y" for i in range(n)])
        t0 = time.monotonic()
        analyze_liquid(code)
        ms = (time.monotonic() - t0) * 1000
        print(f"  {n:3d} functions: {ms:.1f}ms")

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_results_v2.json")
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out}")


if __name__ == "__main__":
    main()
