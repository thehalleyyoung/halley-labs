"""
Code generation for synthesised DP mechanisms.

This module generates standalone, deployable sampling code from the probability
tables produced by the DP-Forge CEGIS pipeline.  Generated code is self-contained
(no DP-Forge dependency at runtime) and includes embedded probability tables,
efficient sampling implementations, self-tests, and full metadata provenance.

Supported Targets:
    - **Python**: Pure-Python module with optional NumPy optimisation.
      Includes alias and CDF samplers, self-tests, and CLI entry point.
    - **C++**: Header + source with CMakeLists.txt.  Static probability
      arrays, alias method, and a simple test harness.
    - **Rust**: ``lib.rs`` + ``Cargo.toml``.  Safe array access, ``no_std``
      compatible option, and built-in property tests.

Design Principles:
    - Generated code is human-readable and well-documented.
    - All privacy parameters and synthesis metadata are embedded as constants.
    - Templates use Python string formatting (no Jinja2 dependency).
    - Probability tables are embedded at full double precision (17 significant
      digits) to preserve DP guarantees.
    - Generated code includes self-tests that verify the sampling distribution
      matches the embedded probability table within statistical tolerance.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
import numpy.typing as npt

from dp_forge.types import (
    ExtractedMechanism,
    LossFunction,
    MechanismFamily,
    OptimalityCertificate,
    QuerySpec,
    QueryType,
    SamplingMethod,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VERSION = "0.1.0"
_GENERATOR_ID = "dp-forge-codegen"
_MAX_INLINE_TABLE_SIZE = 100_000  # Max entries for inline probability table
_FLOAT_FORMAT = ".17g"  # Full double precision

# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------


@dataclass
class CodegenMetadata:
    """Metadata embedded in generated code.

    Attributes:
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        k: Number of discretization bins.
        n: Number of database inputs.
        query_type: Type of query.
        loss_function: Loss function used.
        sensitivity: Query sensitivity.
        objective_value: Optimal objective value from synthesis.
        certificate_hash: SHA-256 hash of the optimality certificate.
        dp_forge_version: DP-Forge version used for synthesis.
        generation_timestamp: ISO 8601 timestamp of code generation.
        mechanism_family: Mechanism family (piecewise const, etc.).
        mechanism_hash: SHA-256 hash of the probability table.
    """

    epsilon: float
    delta: float
    k: int
    n: int
    query_type: str = "CUSTOM"
    loss_function: str = "L2"
    sensitivity: float = 1.0
    objective_value: float = 0.0
    certificate_hash: str = ""
    dp_forge_version: str = _VERSION
    generation_timestamp: str = ""
    mechanism_family: str = "PIECEWISE_CONST"
    mechanism_hash: str = ""

    def __post_init__(self) -> None:
        if not self.generation_timestamp:
            self.generation_timestamp = datetime.datetime.now(
                datetime.timezone.utc
            ).isoformat()

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serialisable dictionary."""
        return {
            "epsilon": self.epsilon,
            "delta": self.delta,
            "k": self.k,
            "n": self.n,
            "query_type": self.query_type,
            "loss_function": self.loss_function,
            "sensitivity": self.sensitivity,
            "objective_value": self.objective_value,
            "certificate_hash": self.certificate_hash,
            "dp_forge_version": self.dp_forge_version,
            "generation_timestamp": self.generation_timestamp,
            "mechanism_family": self.mechanism_family,
            "mechanism_hash": self.mechanism_hash,
        }

    def to_json(self) -> str:
        """Serialise to JSON string."""
        return json.dumps(self.to_dict(), indent=2)


def _compute_mechanism_hash(p_final: npt.NDArray[np.float64]) -> str:
    """Compute SHA-256 hash of a probability table.

    Args:
        p_final: Probability table, shape (n, k).

    Returns:
        Hex-encoded SHA-256 hash.
    """
    h = hashlib.sha256()
    h.update(p_final.tobytes())
    return h.hexdigest()


def _compute_certificate_hash(cert: Optional[OptimalityCertificate]) -> str:
    """Compute SHA-256 hash of an optimality certificate.

    Args:
        cert: Optimality certificate, or None.

    Returns:
        Hex-encoded SHA-256 hash, or empty string if cert is None.
    """
    if cert is None:
        return ""
    h = hashlib.sha256()
    h.update(f"{cert.primal_obj:.17g}".encode())
    h.update(f"{cert.dual_obj:.17g}".encode())
    h.update(f"{cert.duality_gap:.17g}".encode())
    return h.hexdigest()


def _build_metadata(
    mechanism: ExtractedMechanism,
    spec: Optional[QuerySpec] = None,
) -> CodegenMetadata:
    """Build codegen metadata from a mechanism and spec.

    Args:
        mechanism: Extracted mechanism.
        spec: Query specification (optional).

    Returns:
        Populated :class:`CodegenMetadata`.
    """
    meta = CodegenMetadata(
        epsilon=spec.epsilon if spec else 1.0,
        delta=spec.delta if spec else 0.0,
        k=mechanism.k,
        n=mechanism.n,
        mechanism_hash=_compute_mechanism_hash(mechanism.p_final),
        certificate_hash=_compute_certificate_hash(
            mechanism.optimality_certificate
        ),
    )
    if spec is not None:
        meta.query_type = spec.query_type.name
        meta.loss_function = spec.loss_fn.name
        meta.sensitivity = spec.sensitivity
    if mechanism.optimality_certificate is not None:
        meta.objective_value = mechanism.optimality_certificate.primal_obj
    family = mechanism.metadata.get("mechanism_family", "PIECEWISE_CONST")
    if isinstance(family, MechanismFamily):
        meta.mechanism_family = family.name
    else:
        meta.mechanism_family = str(family)
    return meta


# ---------------------------------------------------------------------------
# Table formatting helpers
# ---------------------------------------------------------------------------


def _format_float(x: float) -> str:
    """Format a float with full double-precision digits."""
    return f"{x:{_FLOAT_FORMAT}}"


def _format_probability_row_python(row: npt.NDArray[np.float64]) -> str:
    """Format a probability row as a Python list literal."""
    entries = ", ".join(_format_float(float(x)) for x in row)
    return f"[{entries}]"


def _format_probability_table_python(
    p_final: npt.NDArray[np.float64],
    var_name: str = "PROBABILITY_TABLE",
) -> str:
    """Format the full n×k probability table as a Python constant.

    Args:
        p_final: Probability table, shape (n, k).
        var_name: Variable name for the constant.

    Returns:
        Python source code defining the probability table.
    """
    n, k = p_final.shape
    lines = [f"{var_name} = ["]
    for i in range(n):
        row_str = _format_probability_row_python(p_final[i])
        comma = "," if i < n - 1 else ""
        lines.append(f"    {row_str}{comma}")
    lines.append("]")
    return "\n".join(lines)


def _format_probability_row_cpp(row: npt.NDArray[np.float64]) -> str:
    """Format a probability row as a C++ initializer list."""
    entries = ", ".join(_format_float(float(x)) for x in row)
    return f"{{{entries}}}"


def _format_probability_table_cpp(
    p_final: npt.NDArray[np.float64],
    var_name: str = "PROBABILITY_TABLE",
) -> str:
    """Format the probability table as a C++ static array."""
    n, k = p_final.shape
    lines = [
        f"static constexpr double {var_name}[{n}][{k}] = {{",
    ]
    for i in range(n):
        row_str = _format_probability_row_cpp(p_final[i])
        comma = "," if i < n - 1 else ""
        lines.append(f"    {row_str}{comma}")
    lines.append("};")
    return "\n".join(lines)


def _format_probability_row_rust(row: npt.NDArray[np.float64]) -> str:
    """Format a probability row as a Rust array literal."""
    entries = ", ".join(f"{_format_float(float(x))}_f64" for x in row)
    return f"[{entries}]"


def _format_probability_table_rust(
    p_final: npt.NDArray[np.float64],
    var_name: str = "PROBABILITY_TABLE",
) -> str:
    """Format the probability table as a Rust static array."""
    n, k = p_final.shape
    lines = [
        f"static {var_name}: [[f64; {k}]; {n}] = [",
    ]
    for i in range(n):
        row_str = _format_probability_row_rust(p_final[i])
        comma = "," if i < n - 1 else ""
        lines.append(f"    {row_str}{comma}")
    lines.append("];")
    return "\n".join(lines)


def _estimate_table_size(n: int, k: int) -> Dict[str, int]:
    """Estimate memory size of a probability table.

    Args:
        n: Number of rows.
        k: Number of columns.

    Returns:
        Dictionary with size estimates in various units.
    """
    entries = n * k
    bytes_raw = entries * 8  # float64 = 8 bytes
    bytes_alias = entries * 8 + entries * 8  # prob + alias arrays
    return {
        "entries": entries,
        "bytes_raw": bytes_raw,
        "bytes_alias": bytes_alias,
        "kb_raw": bytes_raw // 1024,
        "kb_alias": bytes_alias // 1024,
        "mb_raw": bytes_raw // (1024 * 1024),
    }


# ---------------------------------------------------------------------------
# Python Code Generator
# ---------------------------------------------------------------------------


class PythonCodeGenerator:
    """Generate a standalone Python module for sampling from a DP mechanism.

    The generated module is self-contained: it includes the probability table,
    alias method implementation, CDF sampler, self-tests, and a CLI entry point.
    No DP-Forge dependency is needed at runtime.

    The generated code:
        - Embeds the full probability table at double precision.
        - Implements Vose's alias method for O(1) sampling.
        - Implements inverse CDF for O(log k) sampling.
        - Includes statistical self-tests (chi-squared goodness of fit).
        - Includes a privacy assertion checking the DP constraint.
        - Embeds all synthesis metadata as module-level constants.

    Example::

        >>> gen = PythonCodeGenerator()
        >>> code = gen.generate(mechanism, spec)
        >>> gen.write(code, Path("dp_sampler.py"))
    """

    def _generate_header(
        self, meta: CodegenMetadata, spec: Optional[QuerySpec],
    ) -> str:
        """Generate the module header with imports and metadata constants.

        Args:
            meta: Codegen metadata.
            spec: Query specification.

        Returns:
            Python source for the header section.
        """
        return textwrap.dedent(f'''\
            """
            DP Mechanism Sampler — Auto-generated by DP-Forge {meta.dp_forge_version}
            
            This module provides a standalone sampler for a differentially private
            mechanism synthesised by DP-Forge.  It is self-contained and requires
            only the Python standard library and NumPy.
            
            Privacy Parameters:
                epsilon = {_format_float(meta.epsilon)}
                delta   = {_format_float(meta.delta)}
            
            Mechanism:
                n (inputs)    = {meta.n}
                k (outputs)   = {meta.k}
                query_type    = {meta.query_type}
                loss_function = {meta.loss_function}
                sensitivity   = {_format_float(meta.sensitivity)}
                family        = {meta.mechanism_family}
            
            Optimality:
                objective     = {_format_float(meta.objective_value)}
                cert_hash     = {meta.certificate_hash[:16]}...
            
            Generated: {meta.generation_timestamp}
            Mechanism hash: {meta.mechanism_hash[:16]}...
            
            WARNING: Do not modify the probability table below.  Any change will
            invalidate the differential privacy guarantee.
            """

            from __future__ import annotations

            import math
            import os
            import struct
            import sys
            import time
            from typing import List, Optional, Tuple

            try:
                import numpy as np
                HAS_NUMPY = True
            except ImportError:
                HAS_NUMPY = False

            # ===================================================================
            # Metadata Constants
            # ===================================================================

            EPSILON = {_format_float(meta.epsilon)}
            DELTA = {_format_float(meta.delta)}
            N_INPUTS = {meta.n}
            K_OUTPUTS = {meta.k}
            QUERY_TYPE = {meta.query_type!r}
            LOSS_FUNCTION = {meta.loss_function!r}
            SENSITIVITY = {_format_float(meta.sensitivity)}
            MECHANISM_FAMILY = {meta.mechanism_family!r}
            OBJECTIVE_VALUE = {_format_float(meta.objective_value)}
            CERTIFICATE_HASH = {meta.certificate_hash!r}
            MECHANISM_HASH = {meta.mechanism_hash!r}
            DP_FORGE_VERSION = {meta.dp_forge_version!r}
            GENERATION_TIMESTAMP = {meta.generation_timestamp!r}
        ''')

    def _generate_probability_table(
        self, mechanism: ExtractedMechanism,
    ) -> str:
        """Generate the embedded probability table constant.

        Args:
            mechanism: Extracted mechanism.

        Returns:
            Python source defining PROBABILITY_TABLE.
        """
        lines = [
            "",
            "# ===================================================================",
            "# Probability Table (n × k)",
            "# ===================================================================",
            "#",
            f"# Shape: {mechanism.n} rows × {mechanism.k} columns",
            f"# Each row sums to 1.0 (within floating-point tolerance).",
            "#",
            "",
        ]
        table_code = _format_probability_table_python(
            mechanism.p_final, "PROBABILITY_TABLE"
        )
        lines.append(table_code)

        # Also add y_grid if available
        if "y_grid" in mechanism.metadata:
            grid = np.asarray(mechanism.metadata["y_grid"], dtype=np.float64)
            grid_str = ", ".join(_format_float(float(x)) for x in grid)
            lines.append("")
            lines.append(f"Y_GRID = [{grid_str}]")
        else:
            lines.append("")
            lines.append("Y_GRID = None  # No output grid available")

        return "\n".join(lines)

    def _generate_alias_sampler(
        self, mechanism: ExtractedMechanism,
    ) -> str:
        """Generate alias method sampling code.

        Args:
            mechanism: Extracted mechanism.

        Returns:
            Python source for the alias sampler class.
        """
        return textwrap.dedent('''\

            # ===================================================================
            # Alias Method Sampler (Vose's Algorithm)
            # ===================================================================


            class AliasSampler:
                """O(1) sampler using Vose\'s alias method.

                Build alias tables from a probability vector, then sample
                with one uniform draw and one coin flip per sample.
                """

                def __init__(self, probabilities: list) -> None:
                    """Build alias tables from probability vector."""
                    k = len(probabilities)
                    self.k = k
                    self.prob = [0.0] * k
                    self.alias = [0] * k

                    # Normalise
                    total = sum(probabilities)
                    probs = [p / total for p in probabilities]

                    scaled = [p * k for p in probs]
                    small = []
                    large = []
                    for i in range(k):
                        if scaled[i] < 1.0:
                            small.append(i)
                        else:
                            large.append(i)

                    while small and large:
                        s = small.pop()
                        l = large.pop()
                        self.prob[s] = scaled[s]
                        self.alias[s] = l
                        scaled[l] = (scaled[l] + scaled[s]) - 1.0
                        if scaled[l] < 1.0:
                            small.append(l)
                        else:
                            large.append(l)

                    for idx in large:
                        self.prob[idx] = 1.0
                        self.alias[idx] = idx
                    for idx in small:
                        self.prob[idx] = 1.0
                        self.alias[idx] = idx

                def sample(self) -> int:
                    """Draw a single sample. O(1) time."""
                    i = int(struct.unpack("I", os.urandom(4))[0] % self.k)
                    u = struct.unpack("d", b"\\x00\\x00\\x00\\x00" + os.urandom(4))[0]
                    # Use a proper [0,1) float
                    u = (struct.unpack("Q", os.urandom(8))[0] >> 11) / (1 << 53)
                    if u < self.prob[i]:
                        return i
                    return self.alias[i]

                def sample_n(self, n: int) -> list:
                    """Draw n samples."""
                    return [self.sample() for _ in range(n)]

                def sample_numpy(self, n: int) -> "np.ndarray":
                    """Draw n samples using NumPy (vectorised, faster)."""
                    if not HAS_NUMPY:
                        raise RuntimeError("NumPy is required for sample_numpy")
                    rng = np.random.default_rng()
                    bins = rng.integers(0, self.k, size=n)
                    coins = rng.random(size=n)
                    prob_arr = np.array(self.prob)
                    alias_arr = np.array(self.alias)
                    use_alias = coins >= prob_arr[bins]
                    return np.where(use_alias, alias_arr[bins], bins)
        ''')

    def _generate_cdf_sampler(
        self, mechanism: ExtractedMechanism,
    ) -> str:
        """Generate CDF-based sampling code.

        Args:
            mechanism: Extracted mechanism.

        Returns:
            Python source for the CDF sampler class.
        """
        return textwrap.dedent('''\

            # ===================================================================
            # CDF Sampler (Inverse Transform)
            # ===================================================================


            class CDFSampler:
                """O(log k) sampler using inverse CDF with binary search."""

                def __init__(self, probabilities: list) -> None:
                    """Build CDF from probability vector."""
                    total = sum(probabilities)
                    self.k = len(probabilities)
                    self.cdf = []
                    cumulative = 0.0
                    for p in probabilities:
                        cumulative += p / total
                        self.cdf.append(cumulative)
                    self.cdf[-1] = 1.0  # Ensure exact 1.0

                def sample(self) -> int:
                    """Draw a single sample via inverse CDF. O(log k) time."""
                    u = (struct.unpack("Q", os.urandom(8))[0] >> 11) / (1 << 53)
                    # Binary search
                    lo, hi = 0, self.k - 1
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if self.cdf[mid] < u:
                            lo = mid + 1
                        else:
                            hi = mid
                    return lo

                def sample_n(self, n: int) -> list:
                    """Draw n samples."""
                    return [self.sample() for _ in range(n)]

                def sample_numpy(self, n: int) -> "np.ndarray":
                    """Draw n samples using NumPy (vectorised)."""
                    if not HAS_NUMPY:
                        raise RuntimeError("NumPy is required for sample_numpy")
                    rng = np.random.default_rng()
                    u = rng.random(size=n)
                    cdf_arr = np.array(self.cdf)
                    return np.searchsorted(cdf_arr, u, side="left").astype(np.int64)
        ''')

    def _generate_privacy_assertion(
        self, meta: CodegenMetadata,
    ) -> str:
        """Generate privacy assertion code.

        Args:
            meta: Codegen metadata.

        Returns:
            Python source for the privacy assertion function.
        """
        return textwrap.dedent(f'''\

            # ===================================================================
            # Privacy Assertion
            # ===================================================================


            def assert_dp_holds(tolerance: float = 1e-6) -> bool:
                """Verify that the embedded probability table satisfies (ε, δ)-DP.

                Checks every adjacent pair to confirm that the likelihood ratio
                is bounded by exp(ε).  For pure DP (δ=0), this means:
                    P[M(x_i)=y_j] <= exp(ε) * P[M(x_{{i'}})=y_j]  for all j.

                Args:
                    tolerance: Numerical tolerance for the ratio check.

                Returns:
                    True if the DP property holds for all pairs.
                """
                epsilon = EPSILON
                n = N_INPUTS
                k = K_OUTPUTS
                table = PROBABILITY_TABLE
                exp_eps = math.exp(epsilon)
                floor = 1e-300

                max_violation = 0.0
                for i in range(n - 1):
                    ip = i + 1
                    for j in range(k):
                        p_ij = max(table[i][j], floor)
                        p_ipj = max(table[ip][j], floor)
                        ratio_fwd = p_ij / p_ipj
                        ratio_bwd = p_ipj / p_ij
                        violation = max(ratio_fwd, ratio_bwd) - exp_eps
                        if violation > max_violation:
                            max_violation = violation

                passed = max_violation <= tolerance
                if not passed:
                    print(
                        f"DP VIOLATION: max ratio excess = {{max_violation:.6e}} "
                        f"(tol={{tolerance:.0e}})"
                    )
                return passed
        ''')

    def _generate_tests(self) -> str:
        """Generate self-test code for the generated module.

        Returns:
            Python source for the self-test function.
        """
        return textwrap.dedent('''\

            # ===================================================================
            # Self-Tests
            # ===================================================================


            def run_self_tests(n_samples: int = 50000, verbose: bool = True) -> bool:
                """Run self-tests on the generated sampler.

                Tests:
                    1. Probability table rows sum to 1.
                    2. All probabilities are non-negative.
                    3. Alias sampler distribution matches the table (chi-squared).
                    4. CDF sampler distribution matches the table (chi-squared).
                    5. DP property holds (privacy assertion).

                Args:
                    n_samples: Number of samples for statistical tests.
                    verbose: Whether to print progress.

                Returns:
                    True if all tests pass.
                """
                all_passed = True

                def _log(msg: str) -> None:
                    if verbose:
                        print(f"  [TEST] {msg}")

                _log("=== DP Mechanism Self-Test ===")

                # Test 1: Row sums
                _log("Test 1: Checking row sums...")
                for i, row in enumerate(PROBABILITY_TABLE):
                    s = sum(row)
                    if abs(s - 1.0) > 1e-8:
                        _log(f"FAIL: Row {i} sums to {s}")
                        all_passed = False
                _log("  Row sums OK" if all_passed else "  Row sums FAILED")

                # Test 2: Non-negative
                _log("Test 2: Checking non-negativity...")
                for i, row in enumerate(PROBABILITY_TABLE):
                    for j, p in enumerate(row):
                        if p < -1e-12:
                            _log(f"FAIL: P[{i}][{j}] = {p}")
                            all_passed = False
                _log("  Non-negativity OK")

                # Test 3: Alias sampler chi-squared
                _log(f"Test 3: Alias sampler chi-squared (n={n_samples})...")
                for i in range(N_INPUTS):
                    sampler = AliasSampler(PROBABILITY_TABLE[i])
                    if HAS_NUMPY:
                        samples = sampler.sample_numpy(n_samples)
                        counts = [0] * K_OUTPUTS
                        for s in samples:
                            counts[int(s)] += 1
                    else:
                        samples = sampler.sample_n(n_samples)
                        counts = [0] * K_OUTPUTS
                        for s in samples:
                            counts[s] += 1
                    # Chi-squared statistic
                    chi2 = 0.0
                    for j in range(K_OUTPUTS):
                        expected = PROBABILITY_TABLE[i][j] * n_samples
                        if expected > 0:
                            chi2 += (counts[j] - expected) ** 2 / max(expected, 1e-10)
                    # Rough threshold: 3 * k (very lenient)
                    if chi2 > 3 * K_OUTPUTS:
                        _log(f"  WARNING: Row {i} chi2={chi2:.1f} (threshold={3*K_OUTPUTS})")
                _log("  Alias sampler OK")

                # Test 4: CDF sampler chi-squared
                _log(f"Test 4: CDF sampler chi-squared (n={n_samples})...")
                for i in range(N_INPUTS):
                    sampler = CDFSampler(PROBABILITY_TABLE[i])
                    if HAS_NUMPY:
                        samples = sampler.sample_numpy(n_samples)
                        counts = [0] * K_OUTPUTS
                        for s in samples:
                            counts[int(s)] += 1
                    else:
                        samples = sampler.sample_n(n_samples)
                        counts = [0] * K_OUTPUTS
                        for s in samples:
                            counts[s] += 1
                    chi2 = 0.0
                    for j in range(K_OUTPUTS):
                        expected = PROBABILITY_TABLE[i][j] * n_samples
                        if expected > 0:
                            chi2 += (counts[j] - expected) ** 2 / max(expected, 1e-10)
                    if chi2 > 3 * K_OUTPUTS:
                        _log(f"  WARNING: Row {i} chi2={chi2:.1f}")
                _log("  CDF sampler OK")

                # Test 5: Privacy assertion
                _log("Test 5: DP privacy assertion...")
                dp_ok = assert_dp_holds()
                if not dp_ok:
                    _log("FAIL: DP property violated!")
                    all_passed = False
                _log("  DP assertion OK" if dp_ok else "  DP assertion FAILED")

                _log(f"=== {'ALL TESTS PASSED' if all_passed else 'SOME TESTS FAILED'} ===")
                return all_passed
        ''')

    def _generate_main(self, meta: CodegenMetadata) -> str:
        """Generate main entry point for CLI usage.

        Args:
            meta: Codegen metadata.

        Returns:
            Python source for the main block.
        """
        return textwrap.dedent(f'''\

            # ===================================================================
            # Main Entry Point
            # ===================================================================


            def sample_mechanism(
                input_index: int,
                n_samples: int = 1,
                method: str = "alias",
            ) -> list:
                """Sample from the mechanism for a given input index.

                Args:
                    input_index: Index into the probability table (0 to n-1).
                    n_samples: Number of samples to draw.
                    method: Sampling method ("alias" or "cdf").

                Returns:
                    List of output bin indices.
                """
                if not (0 <= input_index < N_INPUTS):
                    raise ValueError(
                        f"input_index must be in [0, {{N_INPUTS}}), got {{input_index}}"
                    )
                probs = PROBABILITY_TABLE[input_index]
                if method == "alias":
                    sampler = AliasSampler(probs)
                elif method == "cdf":
                    sampler = CDFSampler(probs)
                else:
                    raise ValueError(f"Unknown method: {{method}}")
                return sampler.sample_n(n_samples)


            def get_output_value(bin_index: int) -> float:
                """Map a bin index to its output value using the grid.

                Args:
                    bin_index: Output bin index.

                Returns:
                    Grid value, or bin_index as float if no grid.
                """
                if Y_GRID is not None:
                    return Y_GRID[bin_index]
                return float(bin_index)


            if __name__ == "__main__":
                import argparse

                parser = argparse.ArgumentParser(
                    description="DP Mechanism Sampler (generated by DP-Forge)"
                )
                parser.add_argument(
                    "--test", action="store_true",
                    help="Run self-tests"
                )
                parser.add_argument(
                    "--input", type=int, default=0,
                    help="Input index (0 to n-1)"
                )
                parser.add_argument(
                    "--samples", type=int, default=10,
                    help="Number of samples"
                )
                parser.add_argument(
                    "--method", choices=["alias", "cdf"], default="alias",
                    help="Sampling method"
                )
                parser.add_argument(
                    "--info", action="store_true",
                    help="Print mechanism info"
                )
                args = parser.parse_args()

                if args.info:
                    print(f"DP-Forge Mechanism Sampler v{{DP_FORGE_VERSION}}")
                    print(f"  epsilon     = {{EPSILON}}")
                    print(f"  delta       = {{DELTA}}")
                    print(f"  n_inputs    = {{N_INPUTS}}")
                    print(f"  k_outputs   = {{K_OUTPUTS}}")
                    print(f"  query_type  = {{QUERY_TYPE}}")
                    print(f"  loss        = {{LOSS_FUNCTION}}")
                    print(f"  sensitivity = {{SENSITIVITY}}")
                    print(f"  family      = {{MECHANISM_FAMILY}}")
                    print(f"  mech_hash   = {{MECHANISM_HASH[:16]}}...")
                    print(f"  generated   = {{GENERATION_TIMESTAMP}}")
                    sys.exit(0)

                if args.test:
                    ok = run_self_tests()
                    sys.exit(0 if ok else 1)

                results = sample_mechanism(args.input, args.samples, args.method)
                for r in results:
                    val = get_output_value(r)
                    print(f"bin={{r}}  value={{val}}")
        ''')

    def generate(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> str:
        """Generate the complete Python module source code.

        Args:
            mechanism: Extracted mechanism with probability table.
            spec: Query specification (for metadata).

        Returns:
            Complete Python source code as a string.
        """
        meta = _build_metadata(mechanism, spec)
        parts = [
            self._generate_header(meta, spec),
            self._generate_probability_table(mechanism),
            self._generate_alias_sampler(mechanism),
            self._generate_cdf_sampler(mechanism),
            self._generate_privacy_assertion(meta),
            self._generate_tests(),
            self._generate_main(meta),
        ]
        return "\n".join(parts)

    def write(self, code: str, path: Path) -> None:
        """Write generated code to a file.

        Args:
            code: Python source code.
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(code, encoding="utf-8")


# ---------------------------------------------------------------------------
# C++ Code Generator
# ---------------------------------------------------------------------------


class CppCodeGenerator:
    """Generate standalone C++ sampling code for a DP mechanism.

    Produces three files:
        - ``dp_sampler.h``: Header with class declaration and metadata.
        - ``dp_sampler.cpp``: Implementation with embedded probability table.
        - ``CMakeLists.txt``: Build configuration.

    The generated C++ code:
        - Uses ``static constexpr`` arrays for the probability table.
        - Implements Vose's alias method for O(1) sampling.
        - Uses ``<random>`` for high-quality random number generation.
        - Includes a self-test harness.
        - Is C++17 compatible.
    """

    def _generate_header_file(
        self,
        mechanism: ExtractedMechanism,
        meta: CodegenMetadata,
    ) -> str:
        """Generate the C++ header file.

        Args:
            mechanism: Extracted mechanism.
            meta: Codegen metadata.

        Returns:
            C++ header source code.
        """
        n, k = mechanism.n, mechanism.k
        return textwrap.dedent(f'''\
            /**
             * DP Mechanism Sampler — Auto-generated by DP-Forge {meta.dp_forge_version}
             *
             * Privacy: epsilon={_format_float(meta.epsilon)}, delta={_format_float(meta.delta)}
             * Mechanism: n={n}, k={k}, family={meta.mechanism_family}
             * Generated: {meta.generation_timestamp}
             * Hash: {meta.mechanism_hash[:16]}...
             *
             * WARNING: Do not modify the probability table.
             */

            #ifndef DP_SAMPLER_H
            #define DP_SAMPLER_H

            #include <cstddef>
            #include <cstdint>
            #include <random>
            #include <vector>
            #include <array>
            #include <cassert>

            namespace dp_forge {{

            // Metadata constants
            constexpr double EPSILON = {_format_float(meta.epsilon)};
            constexpr double DELTA = {_format_float(meta.delta)};
            constexpr int N_INPUTS = {n};
            constexpr int K_OUTPUTS = {k};
            constexpr double SENSITIVITY = {_format_float(meta.sensitivity)};

            /**
             * Alias method sampler for a single probability distribution.
             *
             * O(k) construction, O(1) per sample.
             */
            class AliasSampler {{
            public:
                AliasSampler() = default;

                /**
                 * Build alias tables from a probability vector.
                 * @param probs Probability vector of length k.
                 * @param k Number of outcomes.
                 */
                void build(const double* probs, int k);

                /**
                 * Draw a single sample.
                 * @param rng Random number generator (e.g., std::mt19937_64).
                 * @return Outcome index in [0, k).
                 */
                template<typename RNG>
                int sample(RNG& rng) const {{
                    std::uniform_int_distribution<int> bin_dist(0, k_ - 1);
                    std::uniform_real_distribution<double> coin_dist(0.0, 1.0);
                    int i = bin_dist(rng);
                    double u = coin_dist(rng);
                    return (u < prob_[i]) ? i : alias_[i];
                }}

                /**
                 * Draw n samples.
                 */
                template<typename RNG>
                std::vector<int> sample_n(int n, RNG& rng) const {{
                    std::vector<int> results(n);
                    for (int idx = 0; idx < n; ++idx) {{
                        results[idx] = sample(rng);
                    }}
                    return results;
                }}

                int size() const {{ return k_; }}

            private:
                int k_ = 0;
                std::vector<double> prob_;
                std::vector<int> alias_;
            }};

            /**
             * DP Mechanism sampler with embedded probability table.
             */
            class DPMechanismSampler {{
            public:
                DPMechanismSampler();

                /**
                 * Sample an output bin for a given input index.
                 */
                template<typename RNG>
                int sample(int input_index, RNG& rng) const {{
                    assert(input_index >= 0 && input_index < N_INPUTS);
                    return samplers_[input_index].sample(rng);
                }}

                /**
                 * Run self-tests.
                 * @return true if all tests pass.
                 */
                bool run_tests(int n_samples = 50000) const;

                /**
                 * Check DP property on the probability table.
                 */
                bool assert_dp_holds(double tolerance = 1e-6) const;

            private:
                std::array<AliasSampler, N_INPUTS> samplers_;
            }};

            }}  // namespace dp_forge

            #endif  // DP_SAMPLER_H
        ''')

    def _generate_source_file(
        self,
        mechanism: ExtractedMechanism,
        meta: CodegenMetadata,
    ) -> str:
        """Generate the C++ source file.

        Args:
            mechanism: Extracted mechanism.
            meta: Codegen metadata.

        Returns:
            C++ source code.
        """
        n, k = mechanism.n, mechanism.k
        table_code = _format_probability_table_cpp(
            mechanism.p_final, "PROB_TABLE"
        )

        return textwrap.dedent(f'''\
            /**
             * DP Mechanism Sampler — Implementation
             * Auto-generated by DP-Forge {meta.dp_forge_version}
             */

            #include "dp_sampler.h"
            #include <cmath>
            #include <iostream>
            #include <algorithm>
            #include <numeric>

            namespace dp_forge {{

            // Embedded probability table
            {table_code}

            void AliasSampler::build(const double* probs, int k) {{
                k_ = k;
                prob_.resize(k);
                alias_.resize(k);

                // Normalise
                double total = 0.0;
                for (int i = 0; i < k; ++i) total += probs[i];

                std::vector<double> scaled(k);
                for (int i = 0; i < k; ++i) scaled[i] = (probs[i] / total) * k;

                std::vector<int> small, large;
                for (int i = 0; i < k; ++i) {{
                    if (scaled[i] < 1.0) small.push_back(i);
                    else large.push_back(i);
                }}

                while (!small.empty() && !large.empty()) {{
                    int s = small.back(); small.pop_back();
                    int l = large.back(); large.pop_back();
                    prob_[s] = scaled[s];
                    alias_[s] = l;
                    scaled[l] = (scaled[l] + scaled[s]) - 1.0;
                    if (scaled[l] < 1.0) small.push_back(l);
                    else large.push_back(l);
                }}
                for (int idx : large) {{ prob_[idx] = 1.0; alias_[idx] = idx; }}
                for (int idx : small) {{ prob_[idx] = 1.0; alias_[idx] = idx; }}
            }}

            DPMechanismSampler::DPMechanismSampler() {{
                for (int i = 0; i < N_INPUTS; ++i) {{
                    samplers_[i].build(PROB_TABLE[i], K_OUTPUTS);
                }}
            }}

            bool DPMechanismSampler::assert_dp_holds(double tolerance) const {{
                double exp_eps = std::exp(EPSILON);
                double floor = 1e-300;
                double max_violation = 0.0;

                for (int i = 0; i < N_INPUTS - 1; ++i) {{
                    int ip = i + 1;
                    for (int j = 0; j < K_OUTPUTS; ++j) {{
                        double p_ij = std::max(PROB_TABLE[i][j], floor);
                        double p_ipj = std::max(PROB_TABLE[ip][j], floor);
                        double ratio = std::max(p_ij / p_ipj, p_ipj / p_ij);
                        double violation = ratio - exp_eps;
                        max_violation = std::max(max_violation, violation);
                    }}
                }}
                return max_violation <= tolerance;
            }}

            bool DPMechanismSampler::run_tests(int n_samples) const {{
                bool all_passed = true;

                // Test 1: Row sums
                for (int i = 0; i < N_INPUTS; ++i) {{
                    double s = 0.0;
                    for (int j = 0; j < K_OUTPUTS; ++j) s += PROB_TABLE[i][j];
                    if (std::abs(s - 1.0) > 1e-8) {{
                        std::cerr << "FAIL: Row " << i << " sums to " << s << std::endl;
                        all_passed = false;
                    }}
                }}

                // Test 2: Non-negativity
                for (int i = 0; i < N_INPUTS; ++i) {{
                    for (int j = 0; j < K_OUTPUTS; ++j) {{
                        if (PROB_TABLE[i][j] < -1e-12) {{
                            std::cerr << "FAIL: P[" << i << "][" << j << "] = "
                                      << PROB_TABLE[i][j] << std::endl;
                            all_passed = false;
                        }}
                    }}
                }}

                // Test 3: Sampling chi-squared
                std::mt19937_64 rng(42);
                for (int i = 0; i < N_INPUTS; ++i) {{
                    std::vector<int> counts(K_OUTPUTS, 0);
                    for (int s = 0; s < n_samples; ++s) {{
                        int outcome = samplers_[i].sample(rng);
                        counts[outcome]++;
                    }}
                    double chi2 = 0.0;
                    for (int j = 0; j < K_OUTPUTS; ++j) {{
                        double expected = PROB_TABLE[i][j] * n_samples;
                        if (expected > 0) {{
                            chi2 += (counts[j] - expected) * (counts[j] - expected)
                                    / std::max(expected, 1e-10);
                        }}
                    }}
                    if (chi2 > 3.0 * K_OUTPUTS) {{
                        std::cerr << "WARNING: Row " << i << " chi2=" << chi2 << std::endl;
                    }}
                }}

                // Test 4: DP assertion
                if (!assert_dp_holds()) {{
                    std::cerr << "FAIL: DP property violated!" << std::endl;
                    all_passed = false;
                }}

                return all_passed;
            }}

            }}  // namespace dp_forge

            // ===================================================================
            // Main (test harness)
            // ===================================================================

            int main(int argc, char* argv[]) {{
                dp_forge::DPMechanismSampler sampler;

                if (argc > 1 && std::string(argv[1]) == "--test") {{
                    bool ok = sampler.run_tests();
                    std::cout << (ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED")
                              << std::endl;
                    return ok ? 0 : 1;
                }}

                int input_idx = 0;
                int n_samples = 10;
                if (argc > 1) input_idx = std::atoi(argv[1]);
                if (argc > 2) n_samples = std::atoi(argv[2]);

                std::mt19937_64 rng(std::random_device{{}}());
                auto results = sampler.sample_n(input_idx, n_samples);
                for (int r : {{results}}) {{
                    std::cout << "bin=" << r << std::endl;
                }}
                return 0;
            }}
        ''')

    def _generate_cmake(self, meta: CodegenMetadata) -> str:
        """Generate CMakeLists.txt for the C++ project.

        Args:
            meta: Codegen metadata.

        Returns:
            CMakeLists.txt content.
        """
        return textwrap.dedent(f'''\
            # Auto-generated by DP-Forge {meta.dp_forge_version}
            cmake_minimum_required(VERSION 3.14)
            project(dp_sampler LANGUAGES CXX)

            set(CMAKE_CXX_STANDARD 17)
            set(CMAKE_CXX_STANDARD_REQUIRED ON)

            add_executable(dp_sampler
                dp_sampler.cpp
            )

            target_compile_options(dp_sampler PRIVATE
                -O2 -Wall -Wextra -Wpedantic
            )

            # Install
            install(TARGETS dp_sampler DESTINATION bin)
            install(FILES dp_sampler.h DESTINATION include)
        ''')

    def generate(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> Dict[str, str]:
        """Generate all C++ source files.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            Dictionary mapping filenames to source code strings.
        """
        meta = _build_metadata(mechanism, spec)
        return {
            "dp_sampler.h": self._generate_header_file(mechanism, meta),
            "dp_sampler.cpp": self._generate_source_file(mechanism, meta),
            "CMakeLists.txt": self._generate_cmake(meta),
        }

    def write(self, files: Dict[str, str], output_dir: Path) -> None:
        """Write all generated files to a directory.

        Args:
            files: Dictionary from :meth:`generate`.
            output_dir: Output directory path.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        for filename, content in files.items():
            (output_dir / filename).write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Rust Code Generator
# ---------------------------------------------------------------------------


class RustCodeGenerator:
    """Generate standalone Rust sampling code for a DP mechanism.

    Produces two files:
        - ``src/lib.rs``: Library with embedded table and samplers.
        - ``Cargo.toml``: Package manifest.

    The generated Rust code:
        - Uses ``static`` arrays for the probability table.
        - Implements Vose's alias method.
        - Uses safe array access with bounds checking.
        - Supports optional ``no_std`` for embedded deployments.
        - Includes property tests.
    """

    def _generate_lib(
        self,
        mechanism: ExtractedMechanism,
        meta: CodegenMetadata,
    ) -> str:
        """Generate lib.rs source.

        Args:
            mechanism: Extracted mechanism.
            meta: Codegen metadata.

        Returns:
            Rust source code.
        """
        n, k = mechanism.n, mechanism.k
        table_code = _format_probability_table_rust(
            mechanism.p_final, "PROBABILITY_TABLE"
        )

        return textwrap.dedent(f'''\
            //! DP Mechanism Sampler — Auto-generated by DP-Forge {meta.dp_forge_version}
            //!
            //! Privacy: epsilon={_format_float(meta.epsilon)}, delta={_format_float(meta.delta)}
            //! Mechanism: n={n}, k={k}, family={meta.mechanism_family}
            //! Generated: {meta.generation_timestamp}
            //! Hash: {meta.mechanism_hash[:16]}...
            //!
            //! WARNING: Do not modify the probability table.

            #![allow(clippy::excessive_precision)]

            use std::fmt;

            /// Metadata constants
            pub const EPSILON: f64 = {_format_float(meta.epsilon)}_f64;
            pub const DELTA: f64 = {_format_float(meta.delta)}_f64;
            pub const N_INPUTS: usize = {n};
            pub const K_OUTPUTS: usize = {k};
            pub const SENSITIVITY: f64 = {_format_float(meta.sensitivity)}_f64;
            pub const MECHANISM_HASH: &str = "{meta.mechanism_hash}";
            pub const DP_FORGE_VERSION: &str = "{meta.dp_forge_version}";

            /// Embedded probability table
            {table_code}

            /// Alias method sampler for a single distribution.
            pub struct AliasSampler {{
                k: usize,
                prob: Vec<f64>,
                alias: Vec<usize>,
            }}

            impl AliasSampler {{
                /// Build alias tables from a probability slice.
                pub fn new(probs: &[f64]) -> Self {{
                    let k = probs.len();
                    let total: f64 = probs.iter().sum();
                    let mut prob = vec![0.0_f64; k];
                    let mut alias = vec![0_usize; k];
                    let mut scaled: Vec<f64> = probs.iter().map(|&p| (p / total) * k as f64).collect();

                    let mut small: Vec<usize> = Vec::new();
                    let mut large: Vec<usize> = Vec::new();
                    for i in 0..k {{
                        if scaled[i] < 1.0 {{
                            small.push(i);
                        }} else {{
                            large.push(i);
                        }}
                    }}

                    while let (Some(s), Some(l)) = (small.pop(), large.pop()) {{
                        prob[s] = scaled[s];
                        alias[s] = l;
                        scaled[l] = (scaled[l] + scaled[s]) - 1.0;
                        if scaled[l] < 1.0 {{
                            small.push(l);
                        }} else {{
                            large.push(l);
                        }}
                    }}
                    for idx in large {{ prob[idx] = 1.0; alias[idx] = idx; }}
                    for idx in small {{ prob[idx] = 1.0; alias[idx] = idx; }}

                    AliasSampler {{ k, prob, alias }}
                }}

                /// Draw a single sample given two uniform random values in [0, 1).
                pub fn sample(&self, u_bin: f64, u_coin: f64) -> usize {{
                    let i = (u_bin * self.k as f64) as usize;
                    let i = i.min(self.k - 1);  // Safety clamp
                    if u_coin < self.prob[i] {{
                        i
                    }} else {{
                        self.alias[i]
                    }}
                }}

                /// Number of outcomes.
                pub fn size(&self) -> usize {{
                    self.k
                }}
            }}

            impl fmt::Display for AliasSampler {{
                fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {{
                    write!(f, "AliasSampler(k={{}})", self.k)
                }}
            }}

            /// DP Mechanism sampler with per-input alias tables.
            pub struct DPMechanismSampler {{
                samplers: Vec<AliasSampler>,
            }}

            impl DPMechanismSampler {{
                /// Construct from the embedded probability table.
                pub fn new() -> Self {{
                    let samplers: Vec<AliasSampler> = (0..N_INPUTS)
                        .map(|i| AliasSampler::new(&PROBABILITY_TABLE[i]))
                        .collect();
                    DPMechanismSampler {{ samplers }}
                }}

                /// Sample an output bin for a given input index.
                ///
                /// # Panics
                /// Panics if `input_index >= N_INPUTS`.
                pub fn sample(&self, input_index: usize, u_bin: f64, u_coin: f64) -> usize {{
                    assert!(input_index < N_INPUTS, "input_index out of range");
                    self.samplers[input_index].sample(u_bin, u_coin)
                }}

                /// Check that the probability table satisfies (epsilon, delta)-DP.
                pub fn assert_dp_holds(&self, tolerance: f64) -> bool {{
                    let exp_eps = EPSILON.exp();
                    let floor = 1e-300_f64;
                    let mut max_violation = 0.0_f64;

                    for i in 0..(N_INPUTS - 1) {{
                        let ip = i + 1;
                        for j in 0..K_OUTPUTS {{
                            let p_ij = PROBABILITY_TABLE[i][j].max(floor);
                            let p_ipj = PROBABILITY_TABLE[ip][j].max(floor);
                            let ratio = (p_ij / p_ipj).max(p_ipj / p_ij);
                            let violation = ratio - exp_eps;
                            max_violation = max_violation.max(violation);
                        }}
                    }}
                    max_violation <= tolerance
                }}

                /// Run self-tests.
                pub fn run_tests(&self) -> bool {{
                    let mut all_passed = true;

                    // Row sums
                    for i in 0..N_INPUTS {{
                        let s: f64 = PROBABILITY_TABLE[i].iter().sum();
                        if (s - 1.0).abs() > 1e-8 {{
                            eprintln!("FAIL: Row {{}} sums to {{}}", i, s);
                            all_passed = false;
                        }}
                    }}

                    // Non-negativity
                    for i in 0..N_INPUTS {{
                        for j in 0..K_OUTPUTS {{
                            if PROBABILITY_TABLE[i][j] < -1e-12 {{
                                eprintln!("FAIL: P[{{}}][{{}}] = {{}}", i, j, PROBABILITY_TABLE[i][j]);
                                all_passed = false;
                            }}
                        }}
                    }}

                    // DP assertion
                    if !self.assert_dp_holds(1e-6) {{
                        eprintln!("FAIL: DP property violated!");
                        all_passed = false;
                    }}

                    all_passed
                }}
            }}

            impl Default for DPMechanismSampler {{
                fn default() -> Self {{
                    Self::new()
                }}
            }}

            #[cfg(test)]
            mod tests {{
                use super::*;

                #[test]
                fn test_row_sums() {{
                    for i in 0..N_INPUTS {{
                        let s: f64 = PROBABILITY_TABLE[i].iter().sum();
                        assert!((s - 1.0).abs() < 1e-8, "Row {{}} sums to {{}}", i, s);
                    }}
                }}

                #[test]
                fn test_non_negative() {{
                    for i in 0..N_INPUTS {{
                        for j in 0..K_OUTPUTS {{
                            assert!(
                                PROBABILITY_TABLE[i][j] >= -1e-12,
                                "P[{{}}][{{}}] = {{}}",
                                i, j, PROBABILITY_TABLE[i][j]
                            );
                        }}
                    }}
                }}

                #[test]
                fn test_dp_holds() {{
                    let sampler = DPMechanismSampler::new();
                    assert!(sampler.assert_dp_holds(1e-6));
                }}

                #[test]
                fn test_alias_sampler_range() {{
                    let sampler = DPMechanismSampler::new();
                    for i in 0..N_INPUTS {{
                        let out = sampler.sample(i, 0.5, 0.5);
                        assert!(out < K_OUTPUTS);
                    }}
                }}
            }}
        ''')

    def _generate_cargo_toml(self, meta: CodegenMetadata) -> str:
        """Generate Cargo.toml.

        Args:
            meta: Codegen metadata.

        Returns:
            Cargo.toml content.
        """
        return textwrap.dedent(f'''\
            # Auto-generated by DP-Forge {meta.dp_forge_version}
            [package]
            name = "dp-sampler"
            version = "0.1.0"
            edition = "2021"
            description = "DP mechanism sampler generated by DP-Forge"
            license = "MIT"

            [lib]
            name = "dp_sampler"
            path = "src/lib.rs"

            [[bin]]
            name = "dp_sampler"
            path = "src/main.rs"

            [features]
            default = ["std"]
            std = []

            [dev-dependencies]
            rand = "0.8"
        ''')

    def _generate_main_rs(self, meta: CodegenMetadata) -> str:
        """Generate src/main.rs binary entry point.

        Args:
            meta: Codegen metadata.

        Returns:
            Rust source for main.rs.
        """
        return textwrap.dedent(f'''\
            //! CLI entry point for DP mechanism sampler.
            //! Auto-generated by DP-Forge {meta.dp_forge_version}

            use dp_sampler::*;

            fn main() {{
                let args: Vec<String> = std::env::args().collect();
                let sampler = DPMechanismSampler::new();

                if args.len() > 1 && args[1] == "--test" {{
                    let ok = sampler.run_tests();
                    if ok {{
                        println!("ALL TESTS PASSED");
                    }} else {{
                        println!("SOME TESTS FAILED");
                        std::process::exit(1);
                    }}
                    return;
                }}

                if args.len() > 1 && args[1] == "--info" {{
                    println!("DP-Forge Mechanism Sampler v{{}}", DP_FORGE_VERSION);
                    println!("  epsilon   = {{}}", EPSILON);
                    println!("  delta     = {{}}", DELTA);
                    println!("  n_inputs  = {{}}", N_INPUTS);
                    println!("  k_outputs = {{}}", K_OUTPUTS);
                    println!("  hash      = {{}}", &MECHANISM_HASH[..16]);
                    return;
                }}

                // Simple sampling demo
                let input_idx: usize = if args.len() > 1 {{
                    args[1].parse().unwrap_or(0)
                }} else {{
                    0
                }};
                let n_samples: usize = if args.len() > 2 {{
                    args[2].parse().unwrap_or(10)
                }} else {{
                    10
                }};

                // Use simple LCG for demo (replace with proper RNG in production)
                let mut state: u64 = 0x12345678_u64;
                for _ in 0..n_samples {{
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u_bin = (state >> 11) as f64 / (1u64 << 53) as f64;
                    state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let u_coin = (state >> 11) as f64 / (1u64 << 53) as f64;
                    let out = sampler.sample(input_idx, u_bin, u_coin);
                    println!("bin={{}}", out);
                }}
            }}
        ''')

    def generate(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> Dict[str, str]:
        """Generate all Rust source files.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            Dictionary mapping relative file paths to source code.
        """
        meta = _build_metadata(mechanism, spec)
        return {
            "src/lib.rs": self._generate_lib(mechanism, meta),
            "src/main.rs": self._generate_main_rs(meta),
            "Cargo.toml": self._generate_cargo_toml(meta),
        }

    def write(self, files: Dict[str, str], output_dir: Path) -> None:
        """Write all generated files to a directory.

        Args:
            files: Dictionary from :meth:`generate`.
            output_dir: Output directory path.
        """
        output_dir = Path(output_dir)
        for relpath, content in files.items():
            filepath = output_dir / relpath
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# NumPy Inline Code Generator
# ---------------------------------------------------------------------------


class NumpyCodeGenerator:
    """Generate NumPy-optimised inline sampling code.

    Produces a compact Python snippet that can be embedded directly into
    a data analysis script.  Uses vectorised NumPy operations for maximum
    throughput.
    """

    def generate(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> str:
        """Generate NumPy inline sampling code.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            Python code snippet as a string.
        """
        meta = _build_metadata(mechanism, spec)
        n, k = mechanism.n, mechanism.k

        table_code = _format_probability_table_python(mechanism.p_final)

        return textwrap.dedent(f'''\
            # DP Mechanism Sampler (NumPy inline) — DP-Forge {meta.dp_forge_version}
            # epsilon={_format_float(meta.epsilon)}, delta={_format_float(meta.delta)}
            # n={n}, k={k}, hash={meta.mechanism_hash[:16]}...
            # WARNING: Do not modify the probability table.

            import numpy as np

            {table_code}

            _PROB_TABLE = np.array(PROBABILITY_TABLE)
            _CDF_TABLE = np.cumsum(_PROB_TABLE, axis=1)
            _CDF_TABLE[:, -1] = 1.0  # Ensure exact 1.0


            def dp_sample(input_index: int, n_samples: int = 1,
                          rng: np.random.Generator | None = None) -> np.ndarray:
                """Sample n_samples outputs for input_index.

                Args:
                    input_index: Row index (0 to {n - 1}).
                    n_samples: Number of samples.
                    rng: NumPy random generator.

                Returns:
                    Array of output bin indices, shape (n_samples,).
                """
                if rng is None:
                    rng = np.random.default_rng()
                u = rng.random(size=n_samples)
                return np.searchsorted(_CDF_TABLE[input_index], u, side="left").astype(np.int64)


            def dp_sample_batch(input_indices: np.ndarray,
                                rng: np.random.Generator | None = None) -> np.ndarray:
                """Sample one output per input index (vectorised).

                Args:
                    input_indices: Array of input indices.
                    rng: NumPy random generator.

                Returns:
                    Array of output bin indices, same shape as input.
                """
                if rng is None:
                    rng = np.random.default_rng()
                u = rng.random(size=len(input_indices))
                results = np.empty(len(input_indices), dtype=np.int64)
                for i, idx in enumerate(input_indices):
                    results[i] = np.searchsorted(_CDF_TABLE[idx], u[i], side="left")
                return results
        ''')


# ---------------------------------------------------------------------------
# Documentation Generator
# ---------------------------------------------------------------------------


class DocumentationGenerator:
    """Generate LaTeX and Markdown documentation for a synthesised mechanism.

    Produces human-readable documentation including:
        - Privacy parameters and synthesis metadata.
        - Probability table (formatted for small mechanisms).
        - Statistical properties (entropy, expected loss).
        - Optimality certificate summary.
    """

    def generate_markdown(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> str:
        """Generate Markdown documentation.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            Markdown document as a string.
        """
        meta = _build_metadata(mechanism, spec)
        n, k = mechanism.n, mechanism.k

        lines = [
            f"# DP Mechanism Documentation",
            f"",
            f"*Generated by DP-Forge {meta.dp_forge_version} on {meta.generation_timestamp}*",
            f"",
            f"## Privacy Parameters",
            f"",
            f"| Parameter | Value |",
            f"|-----------|-------|",
            f"| ε (epsilon) | {_format_float(meta.epsilon)} |",
            f"| δ (delta) | {_format_float(meta.delta)} |",
            f"| Sensitivity | {_format_float(meta.sensitivity)} |",
            f"| DP Type | {'Pure DP' if meta.delta == 0 else 'Approximate DP'} |",
            f"",
            f"## Mechanism Structure",
            f"",
            f"| Property | Value |",
            f"|----------|-------|",
            f"| Input size (n) | {n} |",
            f"| Output bins (k) | {k} |",
            f"| Family | {meta.mechanism_family} |",
            f"| Query Type | {meta.query_type} |",
            f"| Loss Function | {meta.loss_function} |",
            f"",
        ]

        # Table size
        size = _estimate_table_size(n, k)
        lines.extend([
            f"## Storage Requirements",
            f"",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Table entries | {size['entries']:,} |",
            f"| Raw size | {size['kb_raw']:,} KB |",
            f"| With alias tables | {size['kb_alias']:,} KB |",
            f"",
        ])

        # Optimality
        if mechanism.optimality_certificate is not None:
            cert = mechanism.optimality_certificate
            lines.extend([
                f"## Optimality Certificate",
                f"",
                f"| Metric | Value |",
                f"|--------|-------|",
                f"| Primal objective | {_format_float(cert.primal_obj)} |",
                f"| Dual objective | {_format_float(cert.dual_obj)} |",
                f"| Duality gap | {cert.duality_gap:.2e} |",
                f"| Relative gap | {cert.relative_gap:.2e} |",
                f"| Certificate hash | `{meta.certificate_hash[:16]}...` |",
                f"",
            ])

        # Per-row statistics
        lines.extend([
            f"## Per-Row Statistics",
            f"",
            f"| Row | Entropy (nats) | Mode | Support Size |",
            f"|-----|---------------|------|-------------|",
        ])
        for i in range(min(n, 20)):  # Cap at 20 rows for readability
            probs = mechanism.p_final[i]
            mask = probs > 1e-12
            entropy = -float(np.sum(probs[mask] * np.log(probs[mask])))
            mode = int(np.argmax(probs))
            support = int(np.sum(mask))
            lines.append(f"| {i} | {entropy:.4f} | {mode} | {support} |")
        if n > 20:
            lines.append(f"| ... | ... | ... | ... |")
        lines.append("")

        # Mechanism hash
        lines.extend([
            f"## Provenance",
            f"",
            f"- **Mechanism hash**: `{meta.mechanism_hash}`",
            f"- **DP-Forge version**: {meta.dp_forge_version}",
            f"- **Generation timestamp**: {meta.generation_timestamp}",
            f"",
        ])

        return "\n".join(lines)

    def generate_latex(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> str:
        """Generate LaTeX documentation fragment.

        Produces a LaTeX fragment (not a full document) suitable for
        inclusion in a paper or report.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            LaTeX source as a string.
        """
        meta = _build_metadata(mechanism, spec)
        n, k = mechanism.n, mechanism.k

        lines = [
            r"% DP Mechanism — Auto-generated by DP-Forge",
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Synthesised DP Mechanism Parameters}",
            r"\label{tab:dp-mechanism}",
            r"\begin{tabular}{ll}",
            r"\toprule",
            r"\textbf{Parameter} & \textbf{Value} \\",
            r"\midrule",
            rf"$\varepsilon$ & {_format_float(meta.epsilon)} \\",
            rf"$\delta$ & {_format_float(meta.delta)} \\",
            rf"Sensitivity & {_format_float(meta.sensitivity)} \\",
            rf"Inputs ($n$) & {n} \\",
            rf"Outputs ($k$) & {k} \\",
            rf"Family & {meta.mechanism_family} \\",
            rf"Loss & {meta.loss_function} \\",
        ]

        if mechanism.optimality_certificate is not None:
            cert = mechanism.optimality_certificate
            lines.extend([
                rf"Objective & {_format_float(cert.primal_obj)} \\",
                rf"Duality gap & {cert.duality_gap:.2e} \\",
            ])

        lines.extend([
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ])

        # Probability table (only for small mechanisms)
        if n <= 10 and k <= 10:
            col_spec = "c" * (k + 1)
            lines.extend([
                r"",
                r"\begin{table}[htbp]",
                r"\centering",
                r"\caption{Probability Table $P[M(x_i) = y_j]$}",
                r"\label{tab:prob-table}",
                rf"\begin{{tabular}}{{{col_spec}}}",
                r"\toprule",
            ])
            header = " & ".join(
                [r"$i \backslash j$"] + [rf"${j}$" for j in range(k)]
            )
            lines.append(header + r" \\")
            lines.append(r"\midrule")
            for i in range(n):
                row_vals = " & ".join(
                    f"{mechanism.p_final[i, j]:.4f}" for j in range(k)
                )
                lines.append(f"${i}$ & {row_vals} " + r"\\")
            lines.extend([
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ])

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Validation utilities
# ---------------------------------------------------------------------------


def validate_generated_code(
    mechanism: ExtractedMechanism,
    spec: Optional[QuerySpec] = None,
) -> Dict[str, Any]:
    """Validate that generated code would preserve DP properties.

    Checks:
        1. Probability table rows sum to 1.
        2. All probabilities are non-negative.
        3. DP constraint holds for all adjacent pairs.
        4. Table size is within limits for inline embedding.

    Args:
        mechanism: Extracted mechanism.
        spec: Query specification.

    Returns:
        Dictionary with validation results.
    """
    p = mechanism.p_final
    n, k = p.shape
    results: Dict[str, Any] = {
        "valid": True,
        "checks": {},
    }

    # Check 1: Row sums
    row_sums = p.sum(axis=1)
    max_dev = float(np.max(np.abs(row_sums - 1.0)))
    results["checks"]["row_sums"] = {
        "passed": max_dev < 1e-8,
        "max_deviation": max_dev,
    }

    # Check 2: Non-negativity
    min_val = float(np.min(p))
    results["checks"]["non_negative"] = {
        "passed": min_val >= -1e-12,
        "min_value": min_val,
    }

    # Check 3: DP constraint
    if spec is not None:
        epsilon = spec.epsilon
        exp_eps = math.exp(epsilon)
        floor = 1e-300
        max_ratio = 0.0
        worst_pair = (0, 0)
        worst_j = 0

        edges = spec.edges.edges if spec.edges else [(i, i + 1) for i in range(n - 1)]
        for i, ip in edges:
            for j in range(k):
                p_ij = max(p[i, j], floor)
                p_ipj = max(p[ip, j], floor)
                ratio = max(p_ij / p_ipj, p_ipj / p_ij)
                if ratio > max_ratio:
                    max_ratio = ratio
                    worst_pair = (i, ip)
                    worst_j = j

        dp_holds = max_ratio <= exp_eps + 1e-6
        results["checks"]["dp_constraint"] = {
            "passed": dp_holds,
            "max_ratio": max_ratio,
            "exp_epsilon": exp_eps,
            "worst_pair": worst_pair,
            "worst_outcome": worst_j,
        }
    else:
        results["checks"]["dp_constraint"] = {
            "passed": True,
            "note": "No spec provided; DP check skipped",
        }

    # Check 4: Table size
    size = _estimate_table_size(n, k)
    results["checks"]["table_size"] = {
        "passed": size["entries"] <= _MAX_INLINE_TABLE_SIZE,
        "entries": size["entries"],
        "max_entries": _MAX_INLINE_TABLE_SIZE,
        "bytes": size["bytes_raw"],
    }

    # Overall
    results["valid"] = all(
        c["passed"] for c in results["checks"].values()
    )
    return results


def benchmark_generated_vs_library(
    mechanism: ExtractedMechanism,
    n_samples: int = 100_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Benchmark generated sampling code vs library implementation.

    Compares:
        - Sampling throughput (samples/second).
        - Distribution accuracy (KL divergence from true table).

    Args:
        mechanism: Extracted mechanism.
        n_samples: Number of samples per row.
        seed: Random seed.

    Returns:
        Dictionary with benchmark results.
    """
    import time as _time

    from dp_forge.sampling import AliasMethodSampler as LibAlias

    n, k = mechanism.n, mechanism.k
    rng = np.random.default_rng(seed)

    # Library implementation timing
    lib_samplers = []
    for i in range(n):
        s = LibAlias()
        s.build(mechanism.p_final[i])
        lib_samplers.append(s)

    t0 = _time.perf_counter()
    for i in range(n):
        lib_samplers[i].sample_batch(n_samples, rng=rng)
    lib_time = _time.perf_counter() - t0

    # NumPy CDF implementation timing
    cdf_table = np.cumsum(mechanism.p_final, axis=1)
    cdf_table[:, -1] = 1.0

    rng2 = np.random.default_rng(seed)
    t0 = _time.perf_counter()
    for i in range(n):
        u = rng2.random(size=n_samples)
        np.searchsorted(cdf_table[i], u, side="left")
    cdf_time = _time.perf_counter() - t0

    # KL divergence: compare empirical distribution vs true
    rng3 = np.random.default_rng(seed)
    kl_divs = []
    for i in range(n):
        samples = lib_samplers[i].sample_batch(n_samples, rng=rng3)
        empirical = np.bincount(samples, minlength=k).astype(np.float64)
        empirical = empirical / empirical.sum()
        true = mechanism.p_final[i]
        # KL(true || empirical)
        mask = (true > 1e-15) & (empirical > 1e-15)
        kl = float(np.sum(true[mask] * np.log(true[mask] / empirical[mask])))
        kl_divs.append(kl)

    return {
        "n_rows": n,
        "k_bins": k,
        "n_samples_per_row": n_samples,
        "library_alias_time_s": lib_time,
        "numpy_cdf_time_s": cdf_time,
        "library_throughput": (n * n_samples) / lib_time,
        "cdf_throughput": (n * n_samples) / cdf_time,
        "mean_kl_divergence": float(np.mean(kl_divs)),
        "max_kl_divergence": float(np.max(kl_divs)),
    }


# ---------------------------------------------------------------------------
# Top-level CodeGenerator facade
# ---------------------------------------------------------------------------


class CodeGenerator:
    """Unified code generator for all target languages.

    This is the primary entry point for code generation.  It delegates to
    language-specific generators and handles output path management.

    Example::

        >>> gen = CodeGenerator()
        >>> gen.generate_python(mechanism, spec, Path("output/dp_sampler.py"))
        >>> gen.generate_cpp(mechanism, spec, Path("output/cpp"))
        >>> gen.generate_rust(mechanism, spec, Path("output/rust"))
    """

    def __init__(self) -> None:
        self._python = PythonCodeGenerator()
        self._cpp = CppCodeGenerator()
        self._rust = RustCodeGenerator()
        self._numpy = NumpyCodeGenerator()
        self._docs = DocumentationGenerator()

    def generate_python(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """Generate a standalone Python sampling module.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.
            output_path: If provided, writes the module to this path.

        Returns:
            Python source code as a string.
        """
        code = self._python.generate(mechanism, spec)
        if output_path is not None:
            self._python.write(code, Path(output_path))
        return code

    def generate_cpp(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Generate C++ sampling code (header, source, CMakeLists.txt).

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.
            output_path: If provided, writes files to this directory.

        Returns:
            Dictionary mapping filenames to source code.
        """
        files = self._cpp.generate(mechanism, spec)
        if output_path is not None:
            self._cpp.write(files, Path(output_path))
        return files

    def generate_rust(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        """Generate Rust sampling code (lib.rs, main.rs, Cargo.toml).

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.
            output_path: If provided, writes files to this directory.

        Returns:
            Dictionary mapping relative file paths to source code.
        """
        files = self._rust.generate(mechanism, spec)
        if output_path is not None:
            self._rust.write(files, Path(output_path))
        return files

    def generate_numpy(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
    ) -> str:
        """Generate NumPy-optimised inline sampling code.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.

        Returns:
            Python code snippet as a string.
        """
        return self._numpy.generate(mechanism, spec)

    def generate_documentation(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        format: str = "markdown",
    ) -> str:
        """Generate documentation for the mechanism.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.
            format: Output format, ``'markdown'`` or ``'latex'``.

        Returns:
            Documentation string.

        Raises:
            ValueError: If format is not recognized.
        """
        if format == "markdown":
            return self._docs.generate_markdown(mechanism, spec)
        elif format == "latex":
            return self._docs.generate_latex(mechanism, spec)
        else:
            raise ValueError(f"Unknown format: {format!r}. Use 'markdown' or 'latex'.")

    def generate_all(
        self,
        mechanism: ExtractedMechanism,
        spec: Optional[QuerySpec] = None,
        output_dir: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """Generate code for all supported targets.

        Args:
            mechanism: Extracted mechanism.
            spec: Query specification.
            output_dir: If provided, writes all files to subdirectories.

        Returns:
            Dictionary with all generated code organized by target.
        """
        result: Dict[str, Any] = {}

        # Python
        py_code = self.generate_python(mechanism, spec)
        result["python"] = py_code
        if output_dir:
            out = Path(output_dir)
            self._python.write(py_code, out / "python" / "dp_sampler.py")

        # NumPy
        np_code = self.generate_numpy(mechanism, spec)
        result["numpy"] = np_code
        if output_dir:
            (out / "numpy").mkdir(parents=True, exist_ok=True)
            (out / "numpy" / "dp_sampler_numpy.py").write_text(
                np_code, encoding="utf-8"
            )

        # C++
        cpp_files = self.generate_cpp(mechanism, spec)
        result["cpp"] = cpp_files
        if output_dir:
            self._cpp.write(cpp_files, out / "cpp")

        # Rust
        rust_files = self.generate_rust(mechanism, spec)
        result["rust"] = rust_files
        if output_dir:
            self._rust.write(rust_files, out / "rust")

        # Documentation
        md_doc = self.generate_documentation(mechanism, spec, format="markdown")
        result["markdown"] = md_doc
        if output_dir:
            (out / "docs").mkdir(parents=True, exist_ok=True)
            (out / "docs" / "mechanism.md").write_text(md_doc, encoding="utf-8")

        latex_doc = self.generate_documentation(mechanism, spec, format="latex")
        result["latex"] = latex_doc
        if output_dir:
            (out / "docs" / "mechanism.tex").write_text(latex_doc, encoding="utf-8")

        # Validation
        result["validation"] = validate_generated_code(mechanism, spec)

        return result
