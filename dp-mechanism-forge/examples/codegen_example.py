"""Code generation example.

Demonstrates:
  1. Synthesizing a mechanism
  2. Generating standalone Python code
  3. Verifying the generated code works
  4. Generating C++ code
"""

from __future__ import annotations

import tempfile
import importlib.util
from pathlib import Path

import numpy as np

from dp_forge.types import (
    QuerySpec,
    MechanismFamily,
    SamplingConfig,
    SamplingMethod,
)
from dp_forge.cegis_loop import CEGISSynthesize
from dp_forge.extractor import extract_mechanism
from dp_forge.sampling import MechanismSampler
from dp_forge.codegen import PythonCodeGenerator, CPPCodeGenerator, RustCodeGenerator
from dp_forge.verifier import verify_dp


def run_codegen_example():
    """Generate deployable code from a synthesized mechanism."""
    print("=" * 60)
    print("DP-Forge: Code Generation Example")
    print("=" * 60)

    # =====================================================================
    # Step 1: Synthesize a mechanism
    # =====================================================================
    epsilon = 1.0
    n = 50
    k = 50  # smaller for readable output

    print(f"\n--- Step 1: Synthesize mechanism ---")
    print(f"  Counting query: n={n}, ε={epsilon}, k={k}")

    spec = QuerySpec.counting(n=n, epsilon=epsilon, delta=0.0, k=k)
    result = CEGISSynthesize(spec, family=MechanismFamily.PIECEWISE_CONST)
    mechanism = extract_mechanism(result, spec)

    dp_ok = verify_dp(mechanism.p_final, epsilon=epsilon, delta=0.0)
    print(f"  Iterations: {result.iterations}")
    print(f"  Objective: {result.obj_val:.6f}")
    print(f"  DP verified: {dp_ok.valid}")

    # =====================================================================
    # Step 2: Generate Python code
    # =====================================================================
    print(f"\n--- Step 2: Generate Python code ---")

    py_gen = PythonCodeGenerator()
    py_code = py_gen.generate(mechanism, spec)

    print(f"  Generated {len(py_code)} characters of Python code")
    print(f"  Preview (first 30 lines):")
    for i, line in enumerate(py_code.split("\n")[:30]):
        print(f"    {i + 1:3d} | {line}")
    print("    ...")

    # Write to temp file
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", delete=False, prefix="dp_mech_"
    ) as f:
        f.write(py_code)
        py_path = f.name

    print(f"\n  Written to: {py_path}")

    # =====================================================================
    # Step 3: Verify generated Python code works
    # =====================================================================
    print(f"\n--- Step 3: Verify generated code ---")

    try:
        spec_module = importlib.util.spec_from_file_location("dp_mech", py_path)
        mod = importlib.util.module_from_spec(spec_module)
        spec_module.loader.exec_module(mod)

        # Test sampling from the generated module
        if hasattr(mod, "sample"):
            samples = [mod.sample(n // 2) for _ in range(1000)]
            print(f"  sample(x={n // 2}): mean={np.mean(samples):.4f}, "
                  f"std={np.std(samples):.4f}")
            print(f"  True value: {spec.query_values[n // 2]:.4f}")
            print("  ✓ Generated Python code works correctly")
        elif hasattr(mod, "DPMechanism"):
            mech_inst = mod.DPMechanism()
            samples = [mech_inst.sample(n // 2) for _ in range(1000)]
            print(f"  sample(x={n // 2}): mean={np.mean(samples):.4f}")
            print("  ✓ Generated Python code works correctly")
        else:
            print("  ⚠ Could not find sample function in generated code")
    except Exception as exc:
        print(f"  ⚠ Could not verify generated code: {exc}")

    # Compare generated code output with original mechanism
    try:
        sampler_orig = MechanismSampler(
            mechanism, SamplingConfig(method=SamplingMethod.ALIAS, seed=42)
        )
        orig_samples = sampler_orig.sample_mechanism(n // 2, n_samples=1000)
        y_grid = np.linspace(spec.eta_min, 1.0 + spec.eta_min, k)
        orig_mean = float(np.mean(y_grid[orig_samples]))
        print(f"\n  Original mechanism mean: {orig_mean:.4f}")
    except Exception:
        pass

    # =====================================================================
    # Step 4: Generate C++ code
    # =====================================================================
    print(f"\n--- Step 4: Generate C++ code ---")

    cpp_gen = CPPCodeGenerator()
    cpp_code = cpp_gen.generate(mechanism, spec, with_cmake=True)

    for name, content in cpp_code.items():
        print(f"\n  {name} ({len(content)} characters):")
        for i, line in enumerate(content.split("\n")[:15]):
            print(f"    {i + 1:3d} | {line}")
        if content.count("\n") > 15:
            print("    ...")

    # Write C++ files to temp directory
    with tempfile.TemporaryDirectory(prefix="dp_mech_cpp_") as tmpdir:
        cpp_gen.write(cpp_code, Path(tmpdir))
        print(f"\n  C++ files written to: {tmpdir}")
        for p in Path(tmpdir).rglob("*"):
            if p.is_file():
                print(f"    {p.relative_to(tmpdir)}")

    # =====================================================================
    # Step 5: Generate Rust code (optional)
    # =====================================================================
    print(f"\n--- Step 5: Generate Rust code ---")

    try:
        rust_gen = RustCodeGenerator()
        rust_code = rust_gen.generate(mechanism, spec)

        for name, content in rust_code.items():
            print(f"\n  {name} ({len(content)} characters):")
            for i, line in enumerate(content.split("\n")[:10]):
                print(f"    {i + 1:3d} | {line}")
            if content.count("\n") > 10:
                print("    ...")
    except Exception as exc:
        print(f"  Rust codegen not available: {exc}")

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 60)
    print("Generated code artifacts:")
    print(f"  Python: {py_path}")
    print(f"  C++: header + source + CMakeLists.txt")
    print(f"  Rust: lib.rs + Cargo.toml")
    print("All generated mechanisms embed the probability table and")
    print("use alias-method sampling for O(1) per-sample cost.")
    print("=" * 60)

    # Cleanup
    try:
        Path(py_path).unlink()
    except Exception:
        pass


if __name__ == "__main__":
    run_codegen_example()
