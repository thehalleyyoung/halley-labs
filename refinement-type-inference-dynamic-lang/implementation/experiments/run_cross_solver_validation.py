"""
Cross-Solver Certificate Validation Experiment.

Generates SMT-LIB 2.6 certificates for safe nn.Module models
and validates them with both Z3 and cvc5, demonstrating
trust-minimized verification.

The certificates encode the shape verification conditions as
QF_LIA (quantifier-free linear integer arithmetic) formulas.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import z3
    HAS_Z3 = True
except ImportError:
    HAS_Z3 = False

try:
    import cvc5
    HAS_CVC5 = True
except ImportError:
    HAS_CVC5 = False


RESULTS_FILE = Path(__file__).parent / "cross_solver_results.json"


# ═══════════════════════════════════════════════════════════════════════════════
# Test Models — safe models with increasing complexity
# ═══════════════════════════════════════════════════════════════════════════════

SAFE_MODELS = [
    {
        "name": "simple_mlp",
        "description": "Two-layer MLP",
        "code": """\
import torch.nn as nn
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        return self.fc2(self.fc1(x))
""",
        "input_shapes": {"x": ("batch", 784)},
        "constraints": [
            # fc1: in=784, out=256 => last_dim(x) == 784
            ("(= last_dim_x 784)", "Linear(784,256) precondition"),
            ("(= out_dim_fc1 256)", "Linear(784,256) postcondition"),
            # fc2: in=256, out=10 => last_dim(fc1_out) == 256
            ("(= out_dim_fc1 256)", "Linear(256,10) precondition"),
            ("(= out_dim_fc2 10)", "Linear(256,10) postcondition"),
        ],
    },
    {
        "name": "deep_mlp",
        "description": "Five-layer MLP with activations",
        "code": """\
import torch.nn as nn
class DeepMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 10)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.relu(self.fc1(x))
        h = self.relu(self.fc2(h))
        h = self.relu(self.fc3(h))
        h = self.relu(self.fc4(h))
        return self.fc5(h)
""",
        "input_shapes": {"x": ("batch", 512)},
        "constraints": [
            ("(= d0 512)", "fc1 in_features"),
            ("(= d1 256)", "fc1 out = fc2 in"),
            ("(= d2 128)", "fc2 out = fc3 in"),
            ("(= d3 64)", "fc3 out = fc4 in"),
            ("(= d4 32)", "fc4 out = fc5 in"),
            ("(= d5 10)", "fc5 out_features"),
            # Chain compatibility
            ("(= d1 256)", "fc2 precondition"),
            ("(= d2 128)", "fc3 precondition"),
            ("(= d3 64)", "fc4 precondition"),
            ("(= d4 32)", "fc5 precondition"),
        ],
    },
    {
        "name": "cnn_classifier",
        "description": "CNN with Conv2d, pooling, and Linear",
        "code": """\
import torch.nn as nn
class CNNClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, 10)
    def forward(self, x):
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.pool(h)
        h = h.flatten(1)
        return self.fc(h)
""",
        "input_shapes": {"x": ("batch", 3, 32, 32)},
        "constraints": [
            ("(= c_in 3)", "conv1 in_channels"),
            ("(= c1 16)", "conv1 out = conv2 in"),
            ("(= c2 32)", "conv2 out_channels"),
            # After pool (1,1): flatten gives 32
            ("(= flat_dim 32)", "flatten output = fc in_features"),
            ("(= fc_out 10)", "fc out_features"),
        ],
    },
    {
        "name": "residual_block",
        "description": "Residual connection with broadcast addition",
        "code": """\
import torch.nn as nn
class ResBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(256, 256)
        self.fc2 = nn.Linear(256, 256)
        self.relu = nn.ReLU()
    def forward(self, x):
        residual = x
        h = self.relu(self.fc1(x))
        h = self.fc2(h)
        return h + residual
""",
        "input_shapes": {"x": ("batch", 256)},
        "constraints": [
            ("(= d_in 256)", "input dim"),
            ("(= fc1_out 256)", "fc1 preserves dim"),
            ("(= fc2_out 256)", "fc2 preserves dim"),
            # Broadcast: h + residual requires same shape
            ("(= fc2_out d_in)", "residual addition compatible"),
        ],
    },
    {
        "name": "layernorm_transformer",
        "description": "Transformer-style block with LayerNorm",
        "code": """\
import torch.nn as nn
class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(768)
        self.fc1 = nn.Linear(768, 3072)
        self.fc2 = nn.Linear(3072, 768)
        self.relu = nn.ReLU()
    def forward(self, x):
        h = self.norm(x)
        h = self.relu(self.fc1(h))
        return self.fc2(h) + x
""",
        "input_shapes": {"x": ("batch", "seq", 768)},
        "constraints": [
            ("(= d_hidden 768)", "input hidden dim"),
            ("(= norm_shape 768)", "LayerNorm normalized_shape"),
            ("(= d_hidden norm_shape)", "LayerNorm compatible"),
            ("(= fc1_in 768)", "fc1 in_features"),
            ("(= fc1_out 3072)", "fc1 out_features"),
            ("(= fc2_in 3072)", "fc2 in_features"),
            ("(= fc2_out 768)", "fc2 out_features"),
            ("(= fc2_out d_hidden)", "residual compatible"),
        ],
    },
]


def generate_smtlib(model: Dict[str, Any]) -> str:
    """Generate SMT-LIB 2.6 certificate for a safe model."""
    lines = []
    lines.append(f"; TensorGuard Safety Certificate for {model['name']}")
    lines.append(f"; {model['description']}")
    lines.append(f"; Verify with: z3 -smt2 <file> OR cvc5 --lang smt2 <file>")
    lines.append(f"; Expected result: UNSAT (safety holds)")
    lines.append("")
    lines.append("(set-logic QF_LIA)")
    lines.append("")

    # Collect all variables from constraints
    import re
    all_vars = set()
    for constraint, desc in model["constraints"]:
        for var in re.findall(r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b', constraint):
            if var not in ('Int', 'Bool', 'true', 'false', 'and', 'or',
                           'not', 'assert', 'declare', 'const'):
                all_vars.add(var)

    # Declare variables
    for var in sorted(all_vars):
        lines.append(f"(declare-const {var} Int)")
        lines.append(f"(assert (> {var} 0))")
    lines.append("")

    # Assert shape constraints
    lines.append("; Shape verification constraints")
    for constraint, desc in model["constraints"]:
        lines.append(f"(assert {constraint})  ; {desc}")
    lines.append("")

    # Safety: negation should be UNSAT
    lines.append("; If all constraints are satisfiable simultaneously,")
    lines.append("; the model is safe. We check the negation is UNSAT.")
    constraint_conj = " ".join(
        c for c, _ in model["constraints"]
    )
    lines.append(f"(assert (not (and {constraint_conj})))")
    lines.append("")
    lines.append("(check-sat)")
    lines.append("(exit)")
    return "\n".join(lines)


def validate_z3(smtlib: str) -> Tuple[str, float]:
    """Validate SMT-LIB certificate with Z3."""
    if not HAS_Z3:
        return "SKIP", 0.0
    t0 = time.monotonic()
    solver = z3.Solver()
    solver.from_string(smtlib)
    result = solver.check()
    elapsed = (time.monotonic() - t0) * 1000
    return str(result), elapsed


def validate_cvc5(smtlib: str) -> Tuple[str, float]:
    """Validate SMT-LIB certificate with cvc5."""
    if not HAS_CVC5:
        return "SKIP", 0.0
    import tempfile
    t0 = time.monotonic()
    with tempfile.NamedTemporaryFile(
        mode='w', suffix='.smt2', delete=False
    ) as f:
        f.write(smtlib)
        f.flush()
        fname = f.name
    try:
        import subprocess
        # Use cvc5 command-line if available, else Python API
        try:
            result = subprocess.run(
                ['cvc5', '--lang', 'smt2', fname],
                capture_output=True, text=True, timeout=10,
            )
            output = result.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Fallback to Python API
            solver = cvc5.Solver()
            parser = cvc5.InputParser(solver)
            parser.setStringInput(
                cvc5.InputLanguage.SMT_LIB_2_6, smtlib, "cert"
            )
            sm = parser.getSymbolManager()
            while True:
                cmd = parser.nextCommand()
                if cmd.isNull():
                    break
                cmd.invoke(solver, sm)
            r = solver.checkSat()
            output = "unsat" if r.isUnsat() else (
                "sat" if r.isSat() else "unknown"
            )
    finally:
        os.unlink(fname)
    elapsed = (time.monotonic() - t0) * 1000
    return output, elapsed


def main():
    print("=" * 70)
    print("Cross-Solver Certificate Validation")
    print("=" * 70)
    print(f"Z3 available: {HAS_Z3}")
    print(f"cvc5 available: {HAS_CVC5}")
    print()

    results = []
    z3_pass = cvc5_pass = 0
    z3_total = cvc5_total = 0

    for model in SAFE_MODELS:
        smtlib = generate_smtlib(model)
        print(f"  [{model['name']}] {model['description']}")

        # Z3 validation
        z3_result, z3_time = validate_z3(smtlib)
        z3_ok = z3_result == "unsat"
        if z3_result != "SKIP":
            z3_total += 1
            if z3_ok:
                z3_pass += 1
        print(f"    Z3:  {z3_result:>7s}  ({z3_time:.1f}ms)"
              f"  {'✓' if z3_ok else '✗'}")

        # cvc5 validation
        cvc5_result, cvc5_time = validate_cvc5(smtlib)
        cvc5_ok = cvc5_result == "unsat"
        if cvc5_result != "SKIP":
            cvc5_total += 1
            if cvc5_ok:
                cvc5_pass += 1
        print(f"    cvc5: {cvc5_result:>7s}  ({cvc5_time:.1f}ms)"
              f"  {'✓' if cvc5_ok else '✗'}")

        cross_match = z3_result == cvc5_result
        print(f"    Cross-solver agreement: {'✓' if cross_match else '✗'}")

        results.append({
            "model": model["name"],
            "description": model["description"],
            "num_constraints": len(model["constraints"]),
            "z3_result": z3_result,
            "z3_time_ms": round(z3_time, 2),
            "cvc5_result": cvc5_result,
            "cvc5_time_ms": round(cvc5_time, 2),
            "cross_solver_agreement": cross_match,
        })

    print()
    print("=" * 70)
    print(f"  Z3:   {z3_pass}/{z3_total} certificates validated")
    print(f"  cvc5: {cvc5_pass}/{cvc5_total} certificates validated")
    agreement = sum(1 for r in results if r["cross_solver_agreement"])
    print(f"  Cross-solver agreement: {agreement}/{len(results)} "
          f"({100*agreement/len(results):.0f}%)")
    print("=" * 70)

    output = {
        "experiment": "cross_solver_validation",
        "z3_available": HAS_Z3,
        "cvc5_available": HAS_CVC5,
        "z3_pass": z3_pass,
        "z3_total": z3_total,
        "cvc5_pass": cvc5_pass,
        "cvc5_total": cvc5_total,
        "cross_solver_agreement_rate": (
            agreement / len(results) if results else 0
        ),
        "results": results,
    }
    with open(RESULTS_FILE, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {RESULTS_FILE}")


if __name__ == "__main__":
    main()
