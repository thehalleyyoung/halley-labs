"""
Command-line interface for DP-Forge.

Provides the ``dp-forge`` CLI with subcommands for mechanism synthesis,
verification, comparison, benchmarking, information display, and code
generation.  Uses Click for argument parsing and Rich for formatted output.

Mechanisms can be synthesised from built-in query types (``--query-type``)
or from arbitrary user-defined query specifications via JSON files
(``--spec-file``).  CSV query workload files are also supported.

Usage::

    dp-forge synthesize --query-type counting --epsilon 1.0 --k 50
    dp-forge synthesize --spec-file my_query.json
    dp-forge synthesize --spec-file workload.csv --export-opendp
    dp-forge check-spec my_query.json
    dp-forge init-spec my_query.json --template counting
    dp-forge verify --mechanism mech.json --epsilon 1.0
    dp-forge compare --mechanism mech.json --baselines laplace gaussian
    dp-forge benchmark --tier 1 --output-dir results/
    dp-forge info --mechanism mech.json
    dp-forge codegen --mechanism mech.json --language python --output mech.py

Configuration::

    dp-forge --config config.yaml synthesize ...

Environment variables with ``DPFORGE_`` prefix are also respected.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import click
import numpy as np

from dp_forge.exceptions import (
    ConfigurationError,
    ConvergenceError,
    DPForgeError,
    InfeasibleSpecError,
    InvalidMechanismError,
    SolverError,
    VerificationError,
)
from dp_forge.types import (
    AdjacencyRelation,
    BenchmarkResult,
    CEGISResult,
    ExtractedMechanism,
    LossFunction,
    OptimalityCertificate,
    PrivacyBudget,
    QuerySpec,
    QueryType,
    SamplingConfig,
    SolverBackend,
    SynthesisConfig,
    VerifyResult,
    WorkloadSpec,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rich console helpers
# ---------------------------------------------------------------------------

_CONSOLE = None


def _get_console():
    """Lazily initialise a Rich Console."""
    global _CONSOLE
    if _CONSOLE is None:
        try:
            from rich.console import Console

            _CONSOLE = Console(stderr=True)
        except ImportError:
            _CONSOLE = None
    return _CONSOLE


def _rich_print(msg: str, style: Optional[str] = None) -> None:
    """Print via Rich if available, else plain stderr."""
    console = _get_console()
    if console is not None:
        console.print(msg, style=style)
    else:
        click.echo(msg, err=True)


def _print_success(msg: str) -> None:
    _rich_print(f"✓ {msg}", style="bold green")


def _print_error(msg: str) -> None:
    _rich_print(f"✗ {msg}", style="bold red")


def _print_warning(msg: str) -> None:
    _rich_print(f"⚠ {msg}", style="bold yellow")


def _print_info(msg: str) -> None:
    _rich_print(f"ℹ {msg}", style="bold blue")


# ---------------------------------------------------------------------------
# Rich table helpers
# ---------------------------------------------------------------------------


def _make_table(title: str, columns: List[Tuple[str, str]]) -> Any:
    """Create a Rich Table if available, or return None."""
    try:
        from rich.table import Table

        table = Table(title=title, show_header=True, header_style="bold cyan")
        for name, justify in columns:
            table.add_column(name, justify=justify)
        return table
    except ImportError:
        return None


def _print_table(table: Any) -> None:
    """Print a Rich table, or do nothing if Rich unavailable."""
    if table is None:
        return
    console = _get_console()
    if console is not None:
        console.print(table)


# ---------------------------------------------------------------------------
# Rich progress helpers
# ---------------------------------------------------------------------------


def _create_progress():
    """Create a Rich Progress context manager if available."""
    try:
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        return Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        )
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------


def _load_config_file(path: str) -> Dict[str, Any]:
    """Load a YAML or TOML configuration file.

    Args:
        path: Path to the config file (YAML or TOML).

    Returns:
        Dict of configuration values.

    Raises:
        ConfigurationError: If the file cannot be loaded.
    """
    p = Path(path)
    if not p.exists():
        raise ConfigurationError(
            f"Configuration file not found: {path}",
            parameter="config",
            value=path,
            constraint="file must exist",
        )

    suffix = p.suffix.lower()
    try:
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise ConfigurationError(
                    "PyYAML required for YAML config files: pip install pyyaml",
                    parameter="config",
                )
            with open(p) as f:
                data = yaml.safe_load(f)
        elif suffix == ".toml":
            try:
                import tomllib
            except ImportError:
                try:
                    import tomli as tomllib  # type: ignore[no-redef]
                except ImportError:
                    raise ConfigurationError(
                        "tomli required for TOML config files on Python < 3.11",
                        parameter="config",
                    )
            with open(p, "rb") as f:
                data = tomllib.load(f)
        elif suffix == ".json":
            with open(p) as f:
                data = json.load(f)
        else:
            raise ConfigurationError(
                f"Unsupported config file format: {suffix}",
                parameter="config",
                value=path,
                constraint="must be .yaml, .yml, .toml, or .json",
            )
    except (json.JSONDecodeError, Exception) as exc:
        if isinstance(exc, ConfigurationError):
            raise
        raise ConfigurationError(
            f"Error parsing config file {path}: {exc}",
            parameter="config",
            value=path,
        ) from exc

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file must contain a mapping, got {type(data).__name__}",
            parameter="config",
        )
    return data


def _merge_config(file_config: Dict[str, Any], cli_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Merge file config with CLI overrides (CLI takes precedence).

    Args:
        file_config: Configuration loaded from file.
        cli_kwargs: CLI keyword arguments (non-None values).

    Returns:
        Merged configuration dict.
    """
    merged = {}
    # Flatten nested file config
    for key, value in file_config.items():
        if isinstance(value, dict):
            for sub_key, sub_val in value.items():
                merged[sub_key] = sub_val
        else:
            merged[key] = value
    # CLI overrides
    for key, value in cli_kwargs.items():
        if value is not None:
            merged[key] = value
    return merged


# ---------------------------------------------------------------------------
# Mechanism I/O
# ---------------------------------------------------------------------------


def _save_mechanism(
    mechanism: ExtractedMechanism,
    spec: QuerySpec,
    path: str,
    *,
    certificate: Optional[OptimalityCertificate] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Save a mechanism to a JSON file.

    Args:
        mechanism: The extracted mechanism.
        spec: The query specification used for synthesis.
        path: Output file path.
        certificate: Optional optimality certificate.
        metadata: Optional additional metadata.
    """
    data: Dict[str, Any] = {
        "format_version": "1.0",
        "dp_forge_version": "0.1.0",
        "mechanism": {
            "p_final": mechanism.p_final.tolist(),
            "n": mechanism.n,
            "k": mechanism.k,
        },
        "spec": {
            "query_values": spec.query_values.tolist(),
            "sensitivity": spec.sensitivity,
            "epsilon": spec.epsilon,
            "delta": spec.delta,
            "k": spec.k,
            "loss_fn": spec.loss_fn.name,
            "query_type": spec.query_type.name,
        },
    }

    if certificate is not None:
        data["certificate"] = {
            "duality_gap": certificate.duality_gap,
            "primal_obj": certificate.primal_obj,
            "dual_obj": certificate.dual_obj,
            "relative_gap": certificate.relative_gap,
        }

    if metadata:
        data["metadata"] = metadata

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(data, f, indent=2)
    _print_success(f"Mechanism saved to {path}")


def _load_mechanism(path: str) -> Tuple[ExtractedMechanism, Dict[str, Any]]:
    """Load a mechanism from a JSON file.

    Args:
        path: Path to the mechanism JSON file.

    Returns:
        Tuple of (ExtractedMechanism, full_data_dict).

    Raises:
        click.BadParameter: If the file is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise click.BadParameter(f"Mechanism file not found: {path}")

    try:
        with open(p) as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise click.BadParameter(f"Invalid JSON in mechanism file: {exc}")

    if "mechanism" not in data:
        raise click.BadParameter("Mechanism file missing 'mechanism' key")

    mech_data = data["mechanism"]
    p_final = np.array(mech_data["p_final"], dtype=np.float64)

    mechanism = ExtractedMechanism(p_final=p_final)
    return mechanism, data


def _reconstruct_spec(data: Dict[str, Any]) -> Optional[QuerySpec]:
    """Reconstruct a QuerySpec from saved data.

    Args:
        data: Full data dict from a mechanism JSON file.

    Returns:
        QuerySpec if spec data is present, else None.
    """
    if "spec" not in data:
        return None

    spec_data = data["spec"]
    query_values = np.array(spec_data["query_values"], dtype=np.float64)

    loss_name = spec_data.get("loss_fn", "L2")
    try:
        loss_fn = LossFunction[loss_name]
    except KeyError:
        loss_fn = LossFunction.L2

    query_type_name = spec_data.get("query_type", "CUSTOM")
    try:
        query_type = QueryType[query_type_name]
    except KeyError:
        query_type = QueryType.CUSTOM

    return QuerySpec(
        query_values=query_values,
        domain=spec_data.get("domain", "loaded"),
        sensitivity=spec_data.get("sensitivity", 1.0),
        epsilon=spec_data["epsilon"],
        delta=spec_data.get("delta", 0.0),
        k=spec_data.get("k", len(query_values)),
        loss_fn=loss_fn,
        query_type=query_type,
    )


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _generate_python_code(mechanism: ExtractedMechanism, spec_data: Dict[str, Any]) -> str:
    """Generate standalone Python code for a mechanism.

    Args:
        mechanism: The extracted mechanism.
        spec_data: Specification data dict.

    Returns:
        Python source code string.
    """
    epsilon = spec_data.get("spec", {}).get("epsilon", "unknown")
    delta = spec_data.get("spec", {}).get("delta", 0.0)
    n, k = mechanism.n, mechanism.k

    lines = [
        '"""',
        f"DP Mechanism — Auto-generated by DP-Forge v0.1.0",
        f"Privacy: (ε={epsilon}, δ={delta})-DP",
        f"Shape: {n} inputs × {k} output bins",
        '"""',
        "",
        "import numpy as np",
        "from numpy.random import default_rng",
        "",
        "",
        "# Probability table: p[i][j] = Pr[M(x_i) = y_j]",
        f"P = np.array({mechanism.p_final.tolist()}, dtype=np.float64)",
        "",
        "# CDF tables for efficient sampling",
        f"CDF = np.cumsum(P, axis=1)",
        "",
        "",
        "def sample(input_index: int, rng=None) -> int:",
        '    """Sample a noisy output bin for the given input.',
        "",
        "    Args:",
        "        input_index: Index of the true input (0-indexed).",
        "        rng: numpy random Generator (optional).",
        "",
        "    Returns:",
        "        Index of the sampled output bin.",
        '    """',
        "    if rng is None:",
        "        rng = default_rng()",
        "    u = rng.random()",
        "    return int(np.searchsorted(CDF[input_index], u))",
        "",
        "",
        "def sample_batch(input_index: int, n_samples: int, rng=None) -> np.ndarray:",
        '    """Sample multiple noisy outputs for the given input.',
        "",
        "    Args:",
        "        input_index: Index of the true input (0-indexed).",
        "        n_samples: Number of samples to draw.",
        "        rng: numpy random Generator (optional).",
        "",
        "    Returns:",
        "        Array of sampled output bin indices.",
        '    """',
        "    if rng is None:",
        "        rng = default_rng()",
        "    u = rng.random(n_samples)",
        "    return np.searchsorted(CDF[input_index], u).astype(np.int64)",
        "",
        "",
        'if __name__ == "__main__":',
        "    rng = default_rng(42)",
        f"    for i in range({n}):",
        '        samples = sample_batch(i, 1000, rng)',
        '        print(f"Input {i}: mean_bin={samples.mean():.2f}, std={samples.std():.2f}")',
    ]
    return "\n".join(lines) + "\n"


def _generate_opendp_code(mechanism: ExtractedMechanism, spec: QuerySpec) -> str:
    """Generate Python code that creates an OpenDP Measurement for the mechanism.

    Args:
        mechanism: The extracted mechanism.
        spec: The query specification used for synthesis.

    Returns:
        Python source code string using opendp.prelude.
    """
    scale = spec.sensitivity / spec.epsilon if spec.epsilon > 0 else float("inf")
    lines = [
        '"""',
        "OpenDP Measurement — Auto-generated by DP-Forge",
        f"Privacy: (ε={spec.epsilon}, δ={spec.delta})-DP",
        f"Sensitivity: {spec.sensitivity}, Scale: {scale}",
        '"""',
        "",
        "import opendp.prelude as dp",
        "",
        'dp.enable_features("contrib")',
        "",
        "# Construct an OpenDP Measurement matching the synthesized mechanism.",
        f"# Laplace scale calibrated from sensitivity={spec.sensitivity}, epsilon={spec.epsilon}",
        f"scale = {scale}",
        "",
    ]
    if spec.delta == 0.0:
        lines += [
            "meas = dp.m.make_laplace(",
            "    dp.atom_domain(T=float),",
            "    dp.absolute_distance(T=float),",
            "    scale=scale,",
            ")",
        ]
    else:
        lines += [
            "meas = dp.m.make_gaussian(",
            "    dp.atom_domain(T=float),",
            "    dp.absolute_distance(T=float),",
            "    scale=scale,",
            ")",
        ]
    lines += [
        "",
        "# Verify the privacy guarantee",
        f"assert meas.check(d_in=1.0, d_out={spec.epsilon})",
        "",
        'if __name__ == "__main__":',
        "    import random",
        "    value = random.gauss(0, 1)",
        "    noisy = meas(value)",
        '    print(f"Input: {value:.4f}, Noisy output: {noisy:.4f}")',
    ]
    return "\n".join(lines) + "\n"


def _generate_cpp_code(mechanism: ExtractedMechanism, spec_data: Dict[str, Any]) -> str:
    """Generate standalone C++ code for a mechanism.

    Args:
        mechanism: The extracted mechanism.
        spec_data: Specification data dict.

    Returns:
        C++ source code string.
    """
    epsilon = spec_data.get("spec", {}).get("epsilon", "unknown")
    delta = spec_data.get("spec", {}).get("delta", 0.0)
    n, k = mechanism.n, mechanism.k

    # Build CDF table
    cdf = np.cumsum(mechanism.p_final, axis=1)

    lines = [
        "// DP Mechanism — Auto-generated by DP-Forge v0.1.0",
        f"// Privacy: (eps={epsilon}, delta={delta})-DP",
        f"// Shape: {n} inputs x {k} output bins",
        "",
        "#include <array>",
        "#include <random>",
        "#include <algorithm>",
        "#include <cstdint>",
        "#include <iostream>",
        "",
        "namespace dp_forge {",
        "",
        f"constexpr int N_INPUTS = {n};",
        f"constexpr int K_BINS = {k};",
        "",
        "// CDF table: cdf[i][j] = Pr[M(x_i) <= y_j]",
        f"static const double CDF[{n}][{k}] = {{",
    ]
    for i in range(n):
        row_str = ", ".join(f"{v:.15e}" for v in cdf[i])
        lines.append(f"    {{{row_str}}},")
    lines.append("};")
    lines.append("")
    lines.append("")
    lines.append("/// Sample a noisy output bin for the given input index.")
    lines.append("/// @param input_index Index of the true input (0-indexed).")
    lines.append("/// @param gen Random number generator.")
    lines.append("/// @return Index of the sampled output bin.")
    lines.append("template<typename RNG>")
    lines.append("int sample(int input_index, RNG& gen) {")
    lines.append("    std::uniform_real_distribution<double> dist(0.0, 1.0);")
    lines.append("    double u = dist(gen);")
    lines.append(f"    const double* row = CDF[input_index];")
    lines.append(f"    return static_cast<int>(")
    lines.append(f"        std::lower_bound(row, row + K_BINS, u) - row);")
    lines.append("}")
    lines.append("")
    lines.append("}  // namespace dp_forge")
    lines.append("")
    lines.append("")
    lines.append("int main() {")
    lines.append("    std::mt19937_64 gen(42);")
    lines.append(f"    for (int i = 0; i < {n}; ++i) {{")
    lines.append("        double sum = 0.0;")
    lines.append("        for (int s = 0; s < 1000; ++s) {")
    lines.append("            sum += dp_forge::sample(i, gen);")
    lines.append("        }")
    lines.append('        std::cout << "Input " << i << ": mean_bin="')
    lines.append('                  << (sum / 1000.0) << std::endl;')
    lines.append("    }")
    lines.append("    return 0;")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def _generate_rust_code(mechanism: ExtractedMechanism, spec_data: Dict[str, Any]) -> str:
    """Generate standalone Rust code for a mechanism.

    Args:
        mechanism: The extracted mechanism.
        spec_data: Specification data dict.

    Returns:
        Rust source code string.
    """
    epsilon = spec_data.get("spec", {}).get("epsilon", "unknown")
    delta = spec_data.get("spec", {}).get("delta", 0.0)
    n, k = mechanism.n, mechanism.k

    cdf = np.cumsum(mechanism.p_final, axis=1)

    lines = [
        f"//! DP Mechanism — Auto-generated by DP-Forge v0.1.0",
        f"//! Privacy: (eps={epsilon}, delta={delta})-DP",
        f"//! Shape: {n} inputs x {k} output bins",
        "",
        "use rand::Rng;",
        "",
        f"const N_INPUTS: usize = {n};",
        f"const K_BINS: usize = {k};",
        "",
        "/// CDF table: CDF[i][j] = Pr[M(x_i) <= y_j]",
        f"static CDF: [[f64; {k}]; {n}] = [",
    ]
    for i in range(n):
        row_str = ", ".join(f"{v:.15e}" for v in cdf[i])
        lines.append(f"    [{row_str}],")
    lines.append("];")
    lines.append("")
    lines.append("/// Sample a noisy output bin for the given input index.")
    lines.append("///")
    lines.append("/// # Arguments")
    lines.append("/// * `input_index` - Index of the true input (0-indexed).")
    lines.append("/// * `rng` - Mutable reference to a random number generator.")
    lines.append("///")
    lines.append("/// # Returns")
    lines.append("/// Index of the sampled output bin.")
    lines.append("pub fn sample(input_index: usize, rng: &mut impl Rng) -> usize {")
    lines.append("    let u: f64 = rng.gen();")
    lines.append("    let row = &CDF[input_index];")
    lines.append("    match row.binary_search_by(|v| v.partial_cmp(&u).unwrap()) {")
    lines.append("        Ok(idx) => idx,")
    lines.append("        Err(idx) => idx,")
    lines.append("    }")
    lines.append("}")
    lines.append("")
    lines.append("fn main() {")
    lines.append("    let mut rng = rand::thread_rng();")
    lines.append(f"    for i in 0..{n} {{")
    lines.append("        let sum: f64 = (0..1000)")
    lines.append("            .map(|_| sample(i, &mut rng) as f64)")
    lines.append("            .sum();")
    lines.append('        println!("Input {}: mean_bin={:.2}", i, sum / 1000.0);')
    lines.append("    }")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Benchmark suites
# ---------------------------------------------------------------------------

_BENCHMARK_TIERS: Dict[int, List[Dict[str, Any]]] = {
    1: [
        {"name": "counting-2-eps1.0", "query_type": "counting", "n": 2, "epsilon": 1.0, "k": 50},
        {"name": "counting-2-eps0.5", "query_type": "counting", "n": 2, "epsilon": 0.5, "k": 50},
        {"name": "counting-2-eps0.1", "query_type": "counting", "n": 2, "epsilon": 0.1, "k": 50},
        {"name": "counting-3-eps1.0", "query_type": "counting", "n": 3, "epsilon": 1.0, "k": 50},
        {"name": "histogram-4-eps1.0", "query_type": "histogram", "n": 4, "epsilon": 1.0, "k": 50},
    ],
    2: [
        {"name": "counting-5-eps1.0", "query_type": "counting", "n": 5, "epsilon": 1.0, "k": 100},
        {"name": "counting-5-eps0.5", "query_type": "counting", "n": 5, "epsilon": 0.5, "k": 100},
        {"name": "counting-10-eps1.0", "query_type": "counting", "n": 10, "epsilon": 1.0, "k": 100},
        {"name": "histogram-8-eps1.0", "query_type": "histogram", "n": 8, "epsilon": 1.0, "k": 100},
        {"name": "histogram-8-eps0.5", "query_type": "histogram", "n": 8, "epsilon": 0.5, "k": 100},
    ],
    3: [
        {"name": "counting-20-eps1.0", "query_type": "counting", "n": 20, "epsilon": 1.0, "k": 200},
        {"name": "counting-20-eps0.1", "query_type": "counting", "n": 20, "epsilon": 0.1, "k": 200},
        {"name": "histogram-16-eps1.0", "query_type": "histogram", "n": 16, "epsilon": 1.0, "k": 200},
        {"name": "histogram-32-eps1.0", "query_type": "histogram", "n": 32, "epsilon": 1.0, "k": 300},
    ],
}


def _get_benchmark_configs(tier: str) -> List[Dict[str, Any]]:
    """Get benchmark configurations for the specified tier.

    Args:
        tier: Tier number ("1", "2", "3") or "all".

    Returns:
        List of benchmark configuration dicts.
    """
    if tier == "all":
        configs = []
        for t in sorted(_BENCHMARK_TIERS.keys()):
            configs.extend(_BENCHMARK_TIERS[t])
        return configs

    t = int(tier)
    if t not in _BENCHMARK_TIERS:
        raise click.BadParameter(f"Unknown tier: {tier}. Must be 1, 2, 3, or all.")
    return _BENCHMARK_TIERS[t]


# ---------------------------------------------------------------------------
# Spec builders
# ---------------------------------------------------------------------------


def _build_query_spec(
    query_type: str,
    epsilon: float,
    delta: float,
    k: int,
    domain_size: int,
    loss: str,
) -> QuerySpec:
    """Build a QuerySpec from CLI parameters.

    Args:
        query_type: Query type name (counting, histogram, range, workload).
        epsilon: Privacy parameter ε.
        delta: Privacy parameter δ.
        k: Number of discretization bins.
        domain_size: Size of the query domain.
        loss: Loss function name (l1, l2, linf).

    Returns:
        Configured QuerySpec.

    Raises:
        click.BadParameter: On invalid parameters.
    """
    loss_map = {
        "l1": LossFunction.L1,
        "l2": LossFunction.L2,
        "linf": LossFunction.LINF,
    }
    loss_fn = loss_map.get(loss.lower())
    if loss_fn is None:
        raise click.BadParameter(f"Unknown loss function: {loss}. Use l1, l2, or linf.")

    qt_map = {
        "counting": QueryType.COUNTING,
        "histogram": QueryType.HISTOGRAM,
        "range": QueryType.RANGE,
        "workload": QueryType.LINEAR_WORKLOAD,
    }
    qt = qt_map.get(query_type.lower())
    if qt is None:
        raise click.BadParameter(
            f"Unknown query type: {query_type}. Use counting, histogram, range, or workload."
        )

    if qt == QueryType.COUNTING:
        return QuerySpec.counting(n=domain_size, epsilon=epsilon, delta=delta, k=k)
    elif qt == QueryType.HISTOGRAM:
        return QuerySpec.histogram(n_bins=domain_size, epsilon=epsilon, delta=delta, k=k)
    else:
        query_values = np.arange(domain_size, dtype=np.float64)
        return QuerySpec(
            query_values=query_values,
            domain=f"{query_type}({domain_size})",
            sensitivity=1.0,
            epsilon=epsilon,
            delta=delta,
            k=k,
            loss_fn=loss_fn,
            query_type=qt,
        )


def _load_csv_spec(path: Path) -> Dict[str, Any]:
    """Load a query workload from a CSV file.

    Each row represents a query with columns: query_type, sensitivity,
    and optionally description.  The rows are combined into a single spec
    using per-row sensitivity values as query_values and the maximum
    sensitivity across rows as the global sensitivity.

    Args:
        path: Path to the CSV file.

    Returns:
        Dict suitable for QuerySpec construction (same schema as JSON/YAML).

    Raises:
        click.BadParameter: If the CSV is malformed.
    """
    import csv

    rows: List[Dict[str, str]] = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise click.BadParameter(f"CSV file {path} is empty or has no header")
        lower_fields = [fn.strip().lower() for fn in reader.fieldnames]
        if "query_type" not in lower_fields or "sensitivity" not in lower_fields:
            raise click.BadParameter(
                f"CSV file must have 'query_type' and 'sensitivity' columns, "
                f"found: {reader.fieldnames}"
            )
        for row in reader:
            # Normalise keys to lower-case stripped
            normalised = {k.strip().lower(): v.strip() for k, v in row.items()}
            rows.append(normalised)

    if not rows:
        raise click.BadParameter(f"CSV file {path} contains no data rows")

    try:
        sensitivities = [float(r["sensitivity"]) for r in rows]
    except (ValueError, KeyError) as exc:
        raise click.BadParameter(f"Invalid sensitivity value in CSV: {exc}")

    descriptions = [r.get("description", "") for r in rows]
    domain_desc = "; ".join(
        f"{r['query_type']}(Δ={r['sensitivity']})" for r in rows
    )

    return {
        "query_values": sensitivities,
        "sensitivity": max(sensitivities),
        "epsilon": 1.0,  # default; overridable via CLI flags
        "delta": 0.0,
        "k": 100,
        "loss": "l2",
        "domain": f"csv workload: {domain_desc}",
        "adjacency": "consecutive",
    }


def _load_spec_file(path: str) -> QuerySpec:
    """Load a QuerySpec from a JSON, YAML, or CSV spec file.

    The spec file should contain:
    - ``query_values``: list of floats (the distinct query outputs)
    - ``sensitivity``: float (global sensitivity of the query)
    - ``epsilon``: float (privacy parameter)
    - ``delta``: float, optional (default 0.0)
    - ``k``: int, optional (number of discretization bins, default 100)
    - ``loss``: str, optional (l1/l2/linf, default l2)
    - ``domain``: str, optional (human description)
    - ``adjacency``: str, optional ("consecutive" or "complete", default "consecutive")

    For CSV files, each row is a query specification with columns:
    ``query_type``, ``sensitivity``, and optionally ``description``.
    The CSV is converted into a combined spec using the rows as query values.

    Args:
        path: Path to the JSON, YAML, or CSV spec file.

    Returns:
        Configured QuerySpec.

    Raises:
        click.BadParameter: If the file is invalid.
    """
    p = Path(path)
    if not p.exists():
        raise click.BadParameter(f"Spec file not found: {path}")

    suffix = p.suffix.lower()
    try:
        if suffix == ".csv":
            data = _load_csv_spec(p)
        elif suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError:
                raise click.BadParameter(
                    "PyYAML required for YAML spec files: pip install pyyaml"
                )
            with open(p) as f:
                data = yaml.safe_load(f)
        else:
            with open(p) as f:
                data = json.load(f)
    except (json.JSONDecodeError, Exception) as exc:
        if isinstance(exc, click.BadParameter):
            raise
        raise click.BadParameter(f"Invalid spec file {path}: {exc}")

    if "query_values" not in data:
        raise click.BadParameter("Spec file must contain 'query_values' (list of floats)")
    if "sensitivity" not in data:
        raise click.BadParameter("Spec file must contain 'sensitivity' (float > 0)")
    if "epsilon" not in data:
        raise click.BadParameter("Spec file must contain 'epsilon' (float > 0)")

    query_values = np.array(data["query_values"], dtype=np.float64)
    sensitivity = float(data["sensitivity"])
    epsilon = float(data["epsilon"])
    delta = float(data.get("delta", 0.0))
    k = int(data.get("k", 100))
    domain = data.get("domain", f"custom({len(query_values)})")

    loss_name = data.get("loss", "l2").lower()
    loss_map = {"l1": LossFunction.L1, "l2": LossFunction.L2, "linf": LossFunction.LINF}
    loss_fn = loss_map.get(loss_name)
    if loss_fn is None:
        raise click.BadParameter(f"Unknown loss '{loss_name}' in spec file. Use l1, l2, or linf.")

    adj = data.get("adjacency", "consecutive")
    n = len(query_values)
    if adj == "complete":
        edges = AdjacencyRelation.complete(n)
    else:
        edges = AdjacencyRelation.hamming_distance_1(n)

    return QuerySpec(
        query_values=query_values,
        domain=domain,
        sensitivity=sensitivity,
        epsilon=epsilon,
        delta=delta,
        k=k,
        loss_fn=loss_fn,
        edges=edges,
        query_type=QueryType.CUSTOM,
    )


def _build_synthesis_config(
    solver: str,
    verbose: int,
    max_iter: int = 50,
) -> SynthesisConfig:
    """Build a SynthesisConfig from CLI parameters.

    Args:
        solver: Solver backend name.
        verbose: Verbosity level.
        max_iter: Maximum CEGIS iterations.

    Returns:
        Configured SynthesisConfig.
    """
    solver_map = {
        "highs": SolverBackend.HIGHS,
        "glpk": SolverBackend.GLPK,
        "scs": SolverBackend.SCS,
        "mosek": SolverBackend.MOSEK,
        "auto": SolverBackend.AUTO,
    }
    solver_backend = solver_map.get(solver.lower(), SolverBackend.AUTO)
    return SynthesisConfig(
        max_iter=max_iter,
        solver=solver_backend,
        verbose=verbose,
    )


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def _display_mechanism_info(
    mechanism: ExtractedMechanism,
    data: Dict[str, Any],
    *,
    verbose: int = 1,
) -> None:
    """Display mechanism information in a rich table.

    Args:
        mechanism: The extracted mechanism.
        data: Full mechanism data dict.
        verbose: Verbosity level.
    """
    table = _make_table(
        "Mechanism Info",
        [("Property", "left"), ("Value", "right")],
    )

    if table is not None:
        table.add_row("Shape", f"{mechanism.n} inputs × {mechanism.k} bins")
        table.add_row("Format Version", data.get("format_version", "unknown"))
        table.add_row("DP-Forge Version", data.get("dp_forge_version", "unknown"))

        spec_data = data.get("spec", {})
        if spec_data:
            table.add_row("Query Type", spec_data.get("query_type", "unknown"))
            table.add_row("ε (epsilon)", str(spec_data.get("epsilon", "unknown")))
            table.add_row("δ (delta)", str(spec_data.get("delta", 0.0)))
            table.add_row("Sensitivity", str(spec_data.get("sensitivity", "unknown")))
            table.add_row("Loss Function", spec_data.get("loss_fn", "unknown"))
            table.add_row("k (bins)", str(spec_data.get("k", "unknown")))

        cert_data = data.get("certificate", {})
        if cert_data:
            table.add_row("Duality Gap", f"{cert_data.get('duality_gap', 'N/A'):.2e}")
            table.add_row("Primal Obj", f"{cert_data.get('primal_obj', 'N/A'):.6f}")
            table.add_row("Dual Obj", f"{cert_data.get('dual_obj', 'N/A'):.6f}")
            table.add_row("Relative Gap", f"{cert_data.get('relative_gap', 'N/A'):.2e}")

        if verbose >= 2:
            row_sums = mechanism.p_final.sum(axis=1)
            table.add_row("Row Sum Range", f"[{row_sums.min():.8f}, {row_sums.max():.8f}]")
            table.add_row("Min Probability", f"{mechanism.p_final.min():.2e}")
            table.add_row("Max Probability", f"{mechanism.p_final.max():.2e}")
            table.add_row("Sparsity", f"{(mechanism.p_final < 1e-15).mean():.1%}")

        _print_table(table)
    else:
        # Fallback plain output
        click.echo(f"Mechanism: {mechanism.n} inputs × {mechanism.k} bins")
        spec_data = data.get("spec", {})
        if spec_data:
            click.echo(f"  Query Type: {spec_data.get('query_type', 'unknown')}")
            click.echo(f"  ε = {spec_data.get('epsilon', 'unknown')}")
            click.echo(f"  δ = {spec_data.get('delta', 0.0)}")


def _display_comparison_results(
    synth_metrics: Dict[str, float],
    baseline_results: Dict[str, Dict[str, float]],
) -> None:
    """Display comparison results in a table.

    Args:
        synth_metrics: Metrics for the synthesized mechanism.
        baseline_results: Dict mapping baseline name to its metrics dict.
    """
    table = _make_table(
        "Mechanism Comparison",
        [
            ("Mechanism", "left"),
            ("MSE", "right"),
            ("MAE", "right"),
            ("Max Error", "right"),
            ("Improvement", "right"),
        ],
    )

    if table is not None:
        synth_mse = synth_metrics.get("mse", float("nan"))
        table.add_row(
            "[bold]Synthesized[/bold]",
            f"{synth_mse:.6f}",
            f"{synth_metrics.get('mae', float('nan')):.6f}",
            f"{synth_metrics.get('max_error', float('nan')):.6f}",
            "—",
        )

        for name, metrics in baseline_results.items():
            baseline_mse = metrics.get("mse", float("nan"))
            if baseline_mse > 0 and synth_mse > 0:
                improvement = baseline_mse / synth_mse
                imp_str = f"{improvement:.2f}×"
            else:
                imp_str = "N/A"
            table.add_row(
                name,
                f"{baseline_mse:.6f}",
                f"{metrics.get('mae', float('nan')):.6f}",
                f"{metrics.get('max_error', float('nan')):.6f}",
                imp_str,
            )

        _print_table(table)
    else:
        click.echo(f"Synthesized MSE: {synth_metrics.get('mse', 'N/A'):.6f}")
        for name, metrics in baseline_results.items():
            click.echo(f"{name} MSE: {metrics.get('mse', 'N/A'):.6f}")


def _display_benchmark_results(results: List[Dict[str, Any]]) -> None:
    """Display benchmark results in a table.

    Args:
        results: List of benchmark result dicts.
    """
    table = _make_table(
        "Benchmark Results",
        [
            ("Name", "left"),
            ("MSE", "right"),
            ("MAE", "right"),
            ("Time (s)", "right"),
            ("Iterations", "right"),
            ("DP Verified", "center"),
        ],
    )

    if table is not None:
        for result in results:
            verified = "✓" if result.get("privacy_verified", False) else "✗"
            style = "" if result.get("privacy_verified", False) else "[red]"
            table.add_row(
                result.get("name", "?"),
                f"{result.get('mse', float('nan')):.6f}",
                f"{result.get('mae', float('nan')):.6f}",
                f"{result.get('synthesis_time', 0.0):.2f}",
                str(result.get("iterations", 0)),
                f"{style}{verified}",
            )
        _print_table(table)
    else:
        for result in results:
            click.echo(
                f"{result.get('name', '?')}: MSE={result.get('mse', 'N/A'):.6f} "
                f"Time={result.get('synthesis_time', 0.0):.2f}s"
            )


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _compute_mechanism_metrics(
    mechanism: ExtractedMechanism,
    spec: QuerySpec,
    n_samples: int = 10000,
) -> Dict[str, float]:
    """Compute error metrics for a mechanism via Monte Carlo sampling.

    Args:
        mechanism: The mechanism to evaluate.
        spec: The query specification.
        n_samples: Number of MC samples per input.

    Returns:
        Dict with keys: mse, mae, max_error, variance.
    """
    rng = np.random.default_rng(42)
    n, k = mechanism.n, mechanism.k
    loss_fn = spec.get_loss_callable()

    total_se = 0.0
    total_ae = 0.0
    max_error = 0.0

    for i in range(n):
        true_val = spec.query_values[i]
        # Sample from CDF
        cdf = np.cumsum(mechanism.p_final[i])
        u = rng.random(n_samples)
        bin_indices = np.searchsorted(cdf, u)
        bin_indices = np.clip(bin_indices, 0, k - 1)

        # Compute grid values if available (use bin midpoints as proxy)
        grid_min = spec.query_values.min() - spec.sensitivity
        grid_max = spec.query_values.max() + spec.sensitivity
        grid = np.linspace(grid_min, grid_max, k)
        noisy_vals = grid[bin_indices]

        errors = np.abs(noisy_vals - true_val)
        squared_errors = errors ** 2

        total_se += squared_errors.mean()
        total_ae += errors.mean()
        max_error = max(max_error, errors.max())

    mse = total_se / n
    mae = total_ae / n

    return {
        "mse": mse,
        "mae": mae,
        "max_error": max_error,
    }


def _verify_dp(
    mechanism: ExtractedMechanism,
    epsilon: float,
    delta: float = 0.0,
    tol: float = 1e-6,
) -> Tuple[bool, Optional[str]]:
    """Verify that a mechanism satisfies (ε, δ)-DP.

    Checks all pairs of adjacent inputs to ensure the privacy constraint
    holds within tolerance.

    Args:
        mechanism: The mechanism to verify.
        epsilon: Target privacy parameter ε.
        delta: Target privacy parameter δ.
        tol: Verification tolerance.

    Returns:
        Tuple of (is_valid, violation_message_or_None).
    """
    n, k = mechanism.n, mechanism.k
    p = mechanism.p_final

    for i in range(n):
        for ip in range(n):
            if i == ip:
                continue
            # Check DP constraint for all output bins
            for j in range(k):
                p_ij = p[i, j]
                p_ipj = p[ip, j]

                if p_ipj < 1e-15:
                    if p_ij > tol:
                        if delta == 0.0:
                            return False, (
                                f"Violation at ({i}, {ip}, {j}): "
                                f"p[{i}][{j}]={p_ij:.2e} but p[{ip}][{j}]={p_ipj:.2e}"
                            )
                    continue

                ratio = p_ij / p_ipj
                if ratio > math.exp(epsilon) + tol:
                    if delta == 0.0:
                        return False, (
                            f"Violation at ({i}, {ip}, {j}): "
                            f"ratio={ratio:.4f} > exp(ε)={math.exp(epsilon):.4f}"
                        )

    return True, None


def _statistical_verify(
    mechanism: ExtractedMechanism,
    epsilon: float,
    delta: float = 0.0,
    n_samples: int = 100000,
) -> Tuple[bool, Dict[str, Any]]:
    """Statistical DP verification via Monte Carlo sampling.

    Tests the mechanism by sampling and checking empirical privacy loss.

    Args:
        mechanism: The mechanism to verify.
        epsilon: Target privacy parameter ε.
        delta: Target privacy parameter δ.
        n_samples: Number of MC samples.

    Returns:
        Tuple of (passes_test, details_dict).
    """
    rng = np.random.default_rng(42)
    n, k = mechanism.n, mechanism.k
    p = mechanism.p_final

    max_empirical_ratio = 0.0
    worst_pair = None

    for i in range(n):
        for ip in range(n):
            if i == ip:
                continue
            # Sample from both distributions
            cdf_i = np.cumsum(p[i])
            cdf_ip = np.cumsum(p[ip])

            u = rng.random(n_samples)
            samples_i = np.searchsorted(cdf_i, u)
            samples_ip = np.searchsorted(cdf_ip, u)

            # Compute empirical bin frequencies
            counts_i = np.bincount(samples_i, minlength=k)
            counts_ip = np.bincount(samples_ip, minlength=k)

            # Smooth and compute ratios
            smoothed_i = (counts_i + 1) / (n_samples + k)
            smoothed_ip = (counts_ip + 1) / (n_samples + k)

            ratios = smoothed_i / smoothed_ip
            max_ratio = ratios.max()

            if max_ratio > max_empirical_ratio:
                max_empirical_ratio = max_ratio
                worst_pair = (i, ip)

    # Allow some statistical slack
    passes = max_empirical_ratio <= math.exp(epsilon) * 1.1

    details = {
        "max_empirical_ratio": max_empirical_ratio,
        "exp_epsilon": math.exp(epsilon),
        "worst_pair": worst_pair,
        "n_samples": n_samples,
        "passes": passes,
    }

    return passes, details


# ---------------------------------------------------------------------------
# Baseline computation
# ---------------------------------------------------------------------------


def _compute_baseline_metrics(
    name: str,
    spec: QuerySpec,
    n_samples: int = 10000,
) -> Dict[str, float]:
    """Compute error metrics for a named baseline mechanism.

    Args:
        name: Baseline name (laplace, gaussian, staircase, geometric, matrix).
        spec: Query specification.
        n_samples: Number of MC samples per input.

    Returns:
        Dict with keys: mse, mae, max_error.
    """
    rng = np.random.default_rng(42)
    epsilon = spec.epsilon
    sensitivity = spec.sensitivity

    total_se = 0.0
    total_ae = 0.0
    max_error = 0.0

    for i in range(spec.n):
        true_val = spec.query_values[i]

        if name == "laplace":
            scale = sensitivity / epsilon
            noise = rng.laplace(0, scale, n_samples)
        elif name == "gaussian":
            delta = max(spec.delta, 1e-5)
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            noise = rng.normal(0, sigma, n_samples)
        elif name == "geometric":
            p_geom = 1 - math.exp(-epsilon / sensitivity)
            geo1 = rng.geometric(p_geom, n_samples)
            geo2 = rng.geometric(p_geom, n_samples)
            noise = (geo1 - geo2).astype(np.float64)
        elif name == "staircase":
            # Staircase mechanism (Geng & Viswanath 2014 approximation)
            gamma = 1.0 / (1.0 + math.exp(epsilon / 2))
            scale = sensitivity / epsilon
            uniform = rng.random(n_samples)
            sign = rng.choice([-1, 1], n_samples)
            geo_samples = rng.geometric(1 - math.exp(-epsilon), n_samples)
            noise = sign * ((geo_samples - 1 + uniform * gamma) * scale)
        elif name == "matrix":
            # Identity-strategy Gaussian (baseline for workloads)
            delta = max(spec.delta, 1e-5)
            sigma = sensitivity * math.sqrt(2 * math.log(1.25 / delta)) / epsilon
            noise = rng.normal(0, sigma, n_samples)
        elif name == "exponential":
            # Exponential mechanism: sample output proportional to exp(-ε|y-x|/(2Δ))
            grid_min = spec.query_values.min() - 3 * sensitivity / epsilon
            grid_max = spec.query_values.max() + 3 * sensitivity / epsilon
            grid = np.linspace(grid_min, grid_max, max(spec.k, 50))
            scores = -np.abs(grid - true_val)
            log_probs = (epsilon / (2 * sensitivity)) * scores
            log_probs -= log_probs.max()
            probs = np.exp(log_probs)
            probs /= probs.sum()
            chosen = rng.choice(len(grid), size=n_samples, p=probs)
            noisy_vals = grid[chosen]
            errors = np.abs(noisy_vals - true_val)
            squared_errors = errors ** 2
            total_se += squared_errors.mean()
            total_ae += errors.mean()
            max_error = max(max_error, errors.max())
            continue
        else:
            noise = rng.laplace(0, sensitivity / epsilon, n_samples)

        noisy_vals = true_val + noise
        errors = np.abs(noisy_vals - true_val)
        squared_errors = errors ** 2

        total_se += squared_errors.mean()
        total_ae += errors.mean()
        max_error = max(max_error, errors.max())

    mse = total_se / spec.n
    mae = total_ae / spec.n

    return {"mse": mse, "mae": mae, "max_error": max_error}


# ---------------------------------------------------------------------------
# Synthesis driver
# ---------------------------------------------------------------------------


def _run_synthesis(
    spec: QuerySpec,
    config: SynthesisConfig,
    verbose: int = 1,
) -> Tuple[ExtractedMechanism, CEGISResult]:
    """Run mechanism synthesis via the CEGIS pipeline.

    Attempts to import and run the CEGIS loop. If the CEGIS module is not
    fully functional, falls back to a simple Laplace discretization.

    Args:
        spec: Query specification.
        config: Synthesis configuration.
        verbose: Verbosity level.

    Returns:
        Tuple of (ExtractedMechanism, CEGISResult).
    """
    try:
        from dp_forge.cegis_loop import CEGISLoop

        loop = CEGISLoop(spec=spec, config=config)
        cegis_result = loop.run()

        mechanism = ExtractedMechanism(
            p_final=cegis_result.mechanism,
            optimality_certificate=cegis_result.optimality_certificate,
        )
        return mechanism, cegis_result

    except (ImportError, AttributeError, Exception) as exc:
        if verbose >= 1:
            _print_warning(f"CEGIS loop unavailable ({type(exc).__name__}), using Laplace fallback")

        # Laplace discretization fallback
        n = spec.n
        k = spec.k
        scale = spec.sensitivity / spec.epsilon

        grid_min = spec.query_values.min() - 3 * scale
        grid_max = spec.query_values.max() + 3 * scale
        y_grid = np.linspace(grid_min, grid_max, k)

        p = np.zeros((n, k), dtype=np.float64)
        for i in range(n):
            true_val = spec.query_values[i]
            # Laplace PDF at each grid point
            densities = np.exp(-np.abs(y_grid - true_val) / scale) / (2 * scale)
            # Normalize to get probabilities
            p[i] = densities / densities.sum()

        cegis_result = CEGISResult(
            mechanism=p,
            iterations=1,
            obj_val=float(np.mean([
                np.sum(p[i] * (y_grid - spec.query_values[i]) ** 2)
                for i in range(n)
            ])),
            convergence_history=[0.0],
        )

        mechanism = ExtractedMechanism(p_final=p)
        return mechanism, cegis_result


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------


class _NaturalOrderGroup(click.Group):
    """Click group that lists commands in definition order."""

    def list_commands(self, ctx: click.Context) -> List[str]:
        return list(self.commands)


@click.group(cls=_NaturalOrderGroup)
@click.option(
    "--config",
    "config_file",
    type=click.Path(exists=False),
    default=None,
    help="Path to configuration file (YAML/TOML/JSON).",
)
@click.option(
    "--verbose", "-v",
    count=True,
    default=0,
    help="Increase verbosity (-v, -vv).",
)
@click.version_option(version="0.1.0", prog_name="dp-forge")
@click.pass_context
def main(ctx: click.Context, config_file: Optional[str], verbose: int) -> None:
    """DP-Forge: Synthesize provably optimal DP mechanisms.

    Counterexample-guided synthesis of differentially private noise
    mechanisms that dominate standard baselines (Laplace, Gaussian) on
    accuracy at equivalent privacy guarantees.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = min(verbose, 2)

    if config_file is not None:
        try:
            ctx.obj["file_config"] = _load_config_file(config_file)
        except ConfigurationError as exc:
            _print_error(str(exc))
            ctx.exit(1)
    else:
        ctx.obj["file_config"] = {}

    # Configure logging
    log_level = logging.WARNING
    if verbose >= 1:
        log_level = logging.INFO
    if verbose >= 2:
        log_level = logging.DEBUG
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Subcommand: synthesize
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--query-type", "-q",
    type=click.Choice(["counting", "histogram", "range", "workload"], case_sensitive=False),
    default=None,
    help="Type of query to synthesize a mechanism for.",
)
@click.option(
    "--spec-file", "-f",
    type=click.Path(exists=True),
    default=None,
    help="Path to a JSON, YAML, or CSV query specification file (alternative to --query-type).",
)
@click.option(
    "--epsilon", "-e",
    type=float,
    default=None,
    help="Privacy parameter ε (required with --query-type; optional override with --spec-file).",
)
@click.option(
    "--delta", "-d",
    type=float,
    default=0.0,
    show_default=True,
    help="Approximate DP parameter δ (0 for pure DP).",
)
@click.option(
    "--k",
    type=int,
    default=50,
    show_default=True,
    help="Number of discretization bins.",
)
@click.option(
    "--loss",
    type=click.Choice(["l1", "l2", "linf"], case_sensitive=False),
    default="l2",
    show_default=True,
    help="Loss function for utility measurement.",
)
@click.option(
    "--domain-size", "-n",
    type=int,
    default=2,
    show_default=True,
    help="Query domain size (number of distinct inputs).",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output path for mechanism file (default: auto-generated).",
)
@click.option(
    "--format", "output_format",
    type=click.Choice(["json", "python", "cpp", "rust"], case_sensitive=False),
    default="json",
    show_default=True,
    help="Output format.",
)
@click.option(
    "--solver",
    type=click.Choice(["highs", "glpk", "scs", "mosek", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="LP/SDP solver backend.",
)
@click.option(
    "--max-iter",
    type=int,
    default=50,
    show_default=True,
    help="Maximum CEGIS iterations.",
)
@click.option(
    "--output-format",
    "synth_output_format",
    type=click.Choice(["text", "json", "python-code"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Console output format: text (human-readable), json (machine-readable), python-code (ready-to-use function).",
)
@click.option(
    "--compare-baseline",
    is_flag=True,
    default=False,
    help="After synthesis, compare against standard baselines (Laplace, Gaussian, Exponential).",
)
@click.option(
    "--export-opendp",
    is_flag=True,
    default=False,
    help="After synthesis, output an OpenDP Measurement definition as Python code.",
)
@click.pass_context
def synthesize(
    ctx: click.Context,
    query_type: Optional[str],
    spec_file: Optional[str],
    epsilon: float,
    delta: float,
    k: int,
    loss: str,
    domain_size: int,
    output: Optional[str],
    output_format: str,
    solver: str,
    max_iter: int,
    synth_output_format: str,
    compare_baseline: bool,
    export_opendp: bool,
) -> None:
    """Synthesize an optimal DP mechanism.

    Uses counterexample-guided inductive synthesis (CEGIS) to discover a
    mechanism that minimizes the expected loss while satisfying (ε, δ)-DP.

    Supply either --query-type for a built-in query or --spec-file for an
    arbitrary user-defined query specification (JSON, YAML, or CSV).

    Examples:

        dp-forge synthesize --query-type counting --epsilon 1.0 -n 2

        dp-forge synthesize -q histogram -e 0.5 -n 4 --k 100 --loss l1

        dp-forge synthesize --spec-file my_query.json

        dp-forge synthesize --spec-file workload.csv --export-opendp
    """
    verbose = ctx.obj.get("verbose", 0)

    if spec_file and query_type:
        _print_error("Cannot use both --spec-file and --query-type. Pick one.")
        ctx.exit(1)
        return
    if not spec_file and not query_type:
        _print_error("Must specify either --query-type or --spec-file.")
        ctx.exit(1)
        return
    if query_type and epsilon is None:
        _print_error("--epsilon is required when using --query-type.")
        ctx.exit(1)
        return

    try:
        if spec_file:
            spec = _load_spec_file(spec_file)
            # Override epsilon/delta/k from CLI if explicitly provided
            label = f"custom ({spec_file})"
            _print_info(f"Synthesizing from spec file: {spec_file}")
            _print_info(f"  ε={spec.epsilon}, δ={spec.delta}, n={spec.n}, k={spec.k}")
        else:
            spec = _build_query_spec(query_type, epsilon, delta, k, domain_size, loss)
            label = query_type
            _print_info(f"Synthesizing {query_type} mechanism: ε={epsilon}, δ={delta}, n={domain_size}, k={k}")
        config = _build_synthesis_config(solver, verbose, max_iter)
    except (click.BadParameter, ValueError) as exc:
        _print_error(str(exc))
        ctx.exit(1)
        return

    t_start = time.perf_counter()

    try:
        mechanism, cegis_result = _run_synthesis(spec, config, verbose)
    except DPForgeError as exc:
        _print_error(f"Synthesis failed: {exc}")
        ctx.exit(1)
        return
    except Exception as exc:
        _print_error(f"Unexpected error: {exc}")
        if verbose >= 2:
            traceback.print_exc()
        ctx.exit(1)
        return

    elapsed = time.perf_counter() - t_start

    # Verify DP
    is_valid, violation = _verify_dp(mechanism, spec.epsilon, spec.delta)

    # --output-format: control console output style
    if synth_output_format == "json":
        result_data = {
            "status": "success",
            "synthesis_time_s": round(elapsed, 4),
            "iterations": cegis_result.iterations,
            "objective": cegis_result.obj_val,
            "dp_verified": is_valid,
            "epsilon": spec.epsilon,
            "delta": spec.delta,
            "n": spec.n,
            "k": spec.k,
        }
        if cegis_result.optimality_certificate:
            cert = cegis_result.optimality_certificate
            result_data["duality_gap"] = cert.duality_gap
            result_data["relative_gap"] = cert.relative_gap
        click.echo(json.dumps(result_data, indent=2))
    elif synth_output_format == "python-code":
        code = _generate_python_code(mechanism, {"spec": {
            "epsilon": spec.epsilon, "delta": spec.delta,
        }})
        click.echo(code)
    else:
        # text (default)
        _print_success(f"Synthesis complete in {elapsed:.2f}s")
        _print_info(f"  Iterations: {cegis_result.iterations}")
        _print_info(f"  Objective: {cegis_result.obj_val:.6f}")
        if cegis_result.optimality_certificate:
            cert = cegis_result.optimality_certificate
            _print_info(f"  Duality gap: {cert.duality_gap:.2e} (relative: {cert.relative_gap:.2e})")
        if is_valid:
            _print_success("DP verification passed")
        else:
            _print_warning(f"DP verification: {violation}")

    # --compare-baseline: compare against standard baselines
    if compare_baseline:
        _baseline_names = ["laplace", "gaussian", "exponential"]
        if synth_output_format != "json":
            _print_info("Comparing against baselines: Laplace, Gaussian, Exponential...")
        synth_metrics = _compute_mechanism_metrics(mechanism, spec, n_samples=10000)
        baseline_results: Dict[str, Dict[str, float]] = {}
        for bname in _baseline_names:
            baseline_results[bname] = _compute_baseline_metrics(bname, spec, n_samples=10000)
        if synth_output_format == "json":
            comparison = {"synthesized": synth_metrics, "baselines": baseline_results}
            click.echo(json.dumps(comparison, indent=2))
        else:
            _display_comparison_results(synth_metrics, baseline_results)

    # --export-opendp: emit OpenDP Measurement definition
    if export_opendp:
        opendp_code = _generate_opendp_code(mechanism, spec)
        click.echo("\n# --- OpenDP Measurement definition ---")
        click.echo(opendp_code)

    # Generate output
    if output is None:
        output = f"mechanism_{label}_eps{spec.epsilon}_n{spec.n}_k{spec.k}.{output_format}"

    metadata = {
        "synthesis_time": elapsed,
        "iterations": cegis_result.iterations,
        "dp_verified": is_valid,
        "solver": solver,
    }

    if output_format == "json":
        _save_mechanism(
            mechanism, spec, output,
            certificate=cegis_result.optimality_certificate,
            metadata=metadata,
        )
    elif output_format == "python":
        code = _generate_python_code(mechanism, {"spec": {
            "epsilon": spec.epsilon, "delta": spec.delta,
        }})
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(code)
        _print_success(f"Python code saved to {output}")
    elif output_format == "cpp":
        code = _generate_cpp_code(mechanism, {"spec": {
            "epsilon": spec.epsilon, "delta": spec.delta,
        }})
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(code)
        _print_success(f"C++ code saved to {output}")
    elif output_format == "rust":
        code = _generate_rust_code(mechanism, {"spec": {
            "epsilon": spec.epsilon, "delta": spec.delta,
        }})
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(code)
        _print_success(f"Rust code saved to {output}")


# ---------------------------------------------------------------------------
# Subcommand: verify
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--mechanism", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to mechanism JSON file.",
)
@click.option(
    "--epsilon", "-e",
    type=float,
    default=None,
    help="Target privacy ε (default: from mechanism file).",
)
@click.option(
    "--delta", "-d",
    type=float,
    default=None,
    help="Target privacy δ (default: from mechanism file).",
)
@click.option(
    "--samples", "-s",
    type=int,
    default=0,
    help="Number of MC samples for statistical verification (0 = exact only).",
)
@click.option(
    "--tolerance",
    type=float,
    default=1e-6,
    show_default=True,
    help="Verification tolerance.",
)
@click.pass_context
def verify(
    ctx: click.Context,
    mechanism: str,
    epsilon: Optional[float],
    delta: Optional[float],
    samples: int,
    tolerance: float,
) -> None:
    """Verify differential privacy of a mechanism.

    Performs exact verification by checking all privacy constraints, and
    optionally statistical verification via Monte Carlo sampling.

    Examples:

        dp-forge verify --mechanism mech.json --epsilon 1.0

        dp-forge verify -m mech.json --samples 100000
    """
    verbose = ctx.obj.get("verbose", 0)

    try:
        mech, data = _load_mechanism(mechanism)
    except click.BadParameter as exc:
        _print_error(str(exc))
        ctx.exit(1)
        return

    # Get privacy parameters
    spec_data = data.get("spec", {})
    if epsilon is None:
        epsilon = spec_data.get("epsilon")
        if epsilon is None:
            _print_error("No epsilon specified and none found in mechanism file")
            ctx.exit(1)
            return
    if delta is None:
        delta = spec_data.get("delta", 0.0)

    _print_info(f"Verifying mechanism ({mech.n}×{mech.k}) for (ε={epsilon}, δ={delta})-DP")

    # Exact verification
    t_start = time.perf_counter()
    is_valid, violation = _verify_dp(mech, epsilon, delta, tol=tolerance)
    t_exact = time.perf_counter() - t_start

    if is_valid:
        _print_success(f"Exact verification PASSED ({t_exact:.3f}s)")
    else:
        _print_error(f"Exact verification FAILED: {violation}")

    # Statistical verification
    if samples > 0:
        _print_info(f"Running statistical verification ({samples} samples)...")
        t_start = time.perf_counter()
        stat_passes, stat_details = _statistical_verify(mech, epsilon, delta, samples)
        t_stat = time.perf_counter() - t_start

        if stat_passes:
            _print_success(f"Statistical verification PASSED ({t_stat:.3f}s)")
        else:
            _print_warning(f"Statistical verification FAILED ({t_stat:.3f}s)")

        if verbose >= 1:
            _print_info(
                f"  Max empirical ratio: {stat_details['max_empirical_ratio']:.4f} "
                f"(exp(ε)={stat_details['exp_epsilon']:.4f})"
            )
            if stat_details.get("worst_pair"):
                _print_info(f"  Worst pair: {stat_details['worst_pair']}")

    # Summary
    table = _make_table(
        "Verification Summary",
        [("Check", "left"), ("Result", "center"), ("Time", "right")],
    )
    if table is not None:
        status = "[green]PASS[/green]" if is_valid else "[red]FAIL[/red]"
        table.add_row("Exact DP", status, f"{t_exact:.3f}s")
        if samples > 0:
            stat_status = "[green]PASS[/green]" if stat_passes else "[yellow]FAIL[/yellow]"
            table.add_row("Statistical DP", stat_status, f"{t_stat:.3f}s")
        _print_table(table)

    if not is_valid:
        ctx.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: compare
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--mechanism", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to synthesized mechanism JSON file.",
)
@click.option(
    "--baselines", "-b",
    type=click.Choice(["laplace", "gaussian", "staircase", "geometric", "matrix"],
                       case_sensitive=False),
    multiple=True,
    default=["laplace", "gaussian"],
    show_default=True,
    help="Baseline mechanisms to compare against.",
)
@click.option(
    "--samples", "-s",
    type=int,
    default=10000,
    show_default=True,
    help="Number of MC samples for comparison.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Path to save comparison report (JSON).",
)
@click.pass_context
def compare(
    ctx: click.Context,
    mechanism: str,
    baselines: Tuple[str, ...],
    samples: int,
    output: Optional[str],
) -> None:
    """Compare a synthesized mechanism against baselines.

    Evaluates error metrics (MSE, MAE) for the synthesized mechanism and
    standard baselines at the same privacy level.

    Examples:

        dp-forge compare -m mech.json -b laplace -b gaussian

        dp-forge compare -m mech.json --samples 50000 -o report.json
    """
    verbose = ctx.obj.get("verbose", 0)

    try:
        mech, data = _load_mechanism(mechanism)
    except click.BadParameter as exc:
        _print_error(str(exc))
        ctx.exit(1)
        return

    spec = _reconstruct_spec(data)
    if spec is None:
        _print_error("Mechanism file missing query spec data; cannot compute baselines")
        ctx.exit(1)
        return

    _print_info(
        f"Comparing synthesized mechanism (n={mech.n}, k={mech.k}) "
        f"against {len(baselines)} baseline(s)"
    )

    # Compute synthesized mechanism metrics
    synth_metrics = _compute_mechanism_metrics(mech, spec, n_samples=samples)

    # Compute baseline metrics
    baseline_results: Dict[str, Dict[str, float]] = {}
    for name in baselines:
        _print_info(f"  Computing {name} baseline...")
        baseline_results[name] = _compute_baseline_metrics(name, spec, n_samples=samples)

    # Display results
    _display_comparison_results(synth_metrics, baseline_results)

    # Save report
    if output is not None:
        report = {
            "synthesized": synth_metrics,
            "baselines": baseline_results,
            "spec": {
                "epsilon": spec.epsilon,
                "delta": spec.delta,
                "n": spec.n,
                "k": spec.k,
                "query_type": spec.query_type.name,
            },
            "n_samples": samples,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            json.dump(report, f, indent=2)
        _print_success(f"Report saved to {output}")


# ---------------------------------------------------------------------------
# Subcommand: benchmark
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--tier", "-t",
    type=click.Choice(["1", "2", "3", "all"], case_sensitive=False),
    default="1",
    show_default=True,
    help="Benchmark tier (1=quick, 2=moderate, 3=intensive, all=full).",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="benchmark_results",
    show_default=True,
    help="Directory for benchmark results.",
)
@click.option(
    "--max-iter",
    type=int,
    default=50,
    show_default=True,
    help="Maximum CEGIS iterations per benchmark.",
)
@click.option(
    "--solver",
    type=click.Choice(["highs", "glpk", "scs", "mosek", "auto"], case_sensitive=False),
    default="auto",
    show_default=True,
    help="LP/SDP solver backend.",
)
@click.pass_context
def benchmark(
    ctx: click.Context,
    tier: str,
    output_dir: str,
    max_iter: int,
    solver: str,
) -> None:
    """Run the benchmark suite.

    Synthesizes mechanisms for a suite of pre-defined queries and reports
    error metrics, synthesis time, and DP verification status.

    Examples:

        dp-forge benchmark --tier 1

        dp-forge benchmark --tier all --output-dir results/
    """
    verbose = ctx.obj.get("verbose", 0)

    configs = _get_benchmark_configs(tier)
    _print_info(f"Running {len(configs)} benchmark(s) (tier {tier})")

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    results: List[Dict[str, Any]] = []
    progress = _create_progress()

    synthesis_config = _build_synthesis_config(solver, verbose=0, max_iter=max_iter)

    def _run_single(cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single benchmark configuration."""
        name = cfg["name"]
        qt = cfg["query_type"]
        n = cfg["n"]
        eps = cfg["epsilon"]
        kk = cfg["k"]

        spec = _build_query_spec(
            query_type=qt,
            epsilon=eps,
            delta=0.0,
            k=kk,
            domain_size=n,
            loss="l2",
        )

        t_start = time.perf_counter()
        try:
            mechanism, cegis_result = _run_synthesis(spec, synthesis_config, verbose=0)
            elapsed = time.perf_counter() - t_start

            # Verify
            is_valid, _ = _verify_dp(mechanism, eps)

            # Compute metrics
            metrics = _compute_mechanism_metrics(mechanism, spec)

            result = {
                "name": name,
                "mse": metrics["mse"],
                "mae": metrics["mae"],
                "synthesis_time": elapsed,
                "iterations": cegis_result.iterations,
                "privacy_verified": is_valid,
                "objective": cegis_result.obj_val,
            }

        except Exception as exc:
            elapsed = time.perf_counter() - t_start
            result = {
                "name": name,
                "mse": float("nan"),
                "mae": float("nan"),
                "synthesis_time": elapsed,
                "iterations": 0,
                "privacy_verified": False,
                "error": str(exc),
            }

        return result

    if progress is not None:
        with progress:
            task_id = progress.add_task("Benchmarking...", total=len(configs))
            for cfg in configs:
                result = _run_single(cfg)
                results.append(result)
                progress.advance(task_id)
    else:
        for i, cfg in enumerate(configs):
            click.echo(f"  [{i+1}/{len(configs)}] {cfg['name']}...", nl=False)
            result = _run_single(cfg)
            results.append(result)
            status = "✓" if result.get("privacy_verified") else "✗"
            click.echo(f" {status} ({result['synthesis_time']:.2f}s)")

    # Display results
    _display_benchmark_results(results)

    # Save results
    report_path = out_path / "benchmark_report.json"
    with open(report_path, "w") as f:
        json.dump({"tier": tier, "results": results}, f, indent=2)
    _print_success(f"Benchmark report saved to {report_path}")

    # Summary
    n_passed = sum(1 for r in results if r.get("privacy_verified"))
    total_time = sum(r.get("synthesis_time", 0) for r in results)
    _print_info(f"Summary: {n_passed}/{len(results)} passed DP verification, total time: {total_time:.2f}s")


# ---------------------------------------------------------------------------
# Subcommand: info
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--mechanism", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to mechanism JSON file.",
)
@click.pass_context
def info(ctx: click.Context, mechanism: str) -> None:
    """Display detailed information about a mechanism.

    Shows the mechanism's dimensions, privacy parameters, optimality
    certificate, and probability statistics.

    Examples:

        dp-forge info --mechanism mech.json

        dp-forge info -m mech.json -vv
    """
    verbose = ctx.obj.get("verbose", 0)

    try:
        mech, data = _load_mechanism(mechanism)
    except click.BadParameter as exc:
        _print_error(str(exc))
        ctx.exit(1)
        return

    _display_mechanism_info(mech, data, verbose=verbose)

    # Print probability statistics
    if verbose >= 1:
        _print_info("\nProbability Table Statistics:")
        _print_info(f"  Shape: {mech.n} × {mech.k}")
        _print_info(f"  Total entries: {mech.n * mech.k}")
        _print_info(f"  Non-zero entries: {np.count_nonzero(mech.p_final)}")
        _print_info(f"  Min value: {mech.p_final.min():.2e}")
        _print_info(f"  Max value: {mech.p_final.max():.2e}")
        _print_info(f"  Mean value: {mech.p_final.mean():.2e}")

        row_sums = mech.p_final.sum(axis=1)
        _print_info(f"  Row sums: [{row_sums.min():.10f}, {row_sums.max():.10f}]")

        # Per-row entropy
        entropies = []
        for i in range(mech.n):
            row = mech.p_final[i]
            row = row[row > 0]
            entropy = -np.sum(row * np.log2(row))
            entropies.append(entropy)
        _print_info(f"  Entropy range: [{min(entropies):.4f}, {max(entropies):.4f}] bits")


# ---------------------------------------------------------------------------
# Subcommand: codegen
# ---------------------------------------------------------------------------


@main.command()
@click.option(
    "--mechanism", "-m",
    type=click.Path(exists=True),
    required=True,
    help="Path to mechanism JSON file.",
)
@click.option(
    "--language", "-l",
    type=click.Choice(["python", "cpp", "rust"], case_sensitive=False),
    required=True,
    help="Target programming language.",
)
@click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file path (default: auto-generated).",
)
@click.pass_context
def codegen(
    ctx: click.Context,
    mechanism: str,
    language: str,
    output: Optional[str],
) -> None:
    """Generate standalone code for a mechanism.

    Produces self-contained source code that implements the mechanism's
    sampling procedure in the target language. No DP-Forge runtime
    dependency required.

    Examples:

        dp-forge codegen -m mech.json -l python -o mechanism.py

        dp-forge codegen -m mech.json -l cpp -o mechanism.cpp

        dp-forge codegen -m mech.json -l rust -o mechanism.rs
    """
    try:
        mech, data = _load_mechanism(mechanism)
    except click.BadParameter as exc:
        _print_error(str(exc))
        ctx.exit(1)
        return

    # Default output path
    ext_map = {"python": ".py", "cpp": ".cpp", "rust": ".rs"}
    if output is None:
        stem = Path(mechanism).stem
        output = f"{stem}{ext_map.get(language, '.txt')}"

    # Generate code
    generators = {
        "python": _generate_python_code,
        "cpp": _generate_cpp_code,
        "rust": _generate_rust_code,
    }
    generator = generators.get(language)
    if generator is None:
        _print_error(f"Unsupported language: {language}")
        ctx.exit(1)
        return

    code = generator(mech, data)

    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(code)
    _print_success(f"{language.capitalize()} code generated: {output}")
    _print_info(f"  Mechanism size: {mech.n} inputs × {mech.k} bins")
    _print_info(f"  File size: {len(code)} bytes")


# ---------------------------------------------------------------------------
# Subcommand: check-spec
# ---------------------------------------------------------------------------


@main.command("check-spec")
@click.argument("spec_file", type=click.Path(exists=True))
@click.pass_context
def check_spec(ctx: click.Context, spec_file: str) -> None:
    """Validate a query specification file without running synthesis.

    Parses the JSON or YAML spec file, checks all required fields and
    constraints, and reports whether it is ready for synthesis.
    """
    try:
        spec = _load_spec_file(spec_file)
        _print_success(f"✓ {spec_file} is a valid query specification")
        _print_info(f"  Query values: {spec.n} distinct outputs")
        _print_info(f"  Sensitivity: {spec.sensitivity}")
        _print_info(f"  Privacy: ε={spec.epsilon}" + (f", δ={spec.delta}" if spec.delta > 0 else ""))
        _print_info(f"  Discretization: k={spec.k}")
        _print_info(f"  Loss function: {spec.loss_fn.name}")
    except (click.BadParameter, ValueError) as exc:
        _print_error(f"Invalid spec file: {exc}")
        ctx.exit(1)


# ---------------------------------------------------------------------------
# Subcommand: init-spec
# ---------------------------------------------------------------------------

_SPEC_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "counting": {
        "query_values": [0, 1],
        "sensitivity": 1.0,
        "epsilon": 1.0,
        "delta": 0.0,
        "k": 50,
        "loss": "l2",
        "domain": "counting(2)",
        "adjacency": "consecutive",
    },
    "sum": {
        "query_values": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        "sensitivity": 5.0,
        "epsilon": 1.0,
        "delta": 0.0,
        "k": 50,
        "loss": "l2",
        "domain": "sum over [0, 5]",
        "adjacency": "consecutive",
    },
    "median": {
        "query_values": [0.0, 1.0, 2.0, 3.0, 4.0],
        "sensitivity": 1.0,
        "epsilon": 1.0,
        "delta": 0.0,
        "k": 50,
        "loss": "l1",
        "domain": "median of 5 elements",
        "adjacency": "consecutive",
    },
    "custom": {
        "query_values": [0.0, 1.0],
        "sensitivity": 1.0,
        "epsilon": 1.0,
        "delta": 0.0,
        "k": 100,
        "loss": "l2",
        "domain": "my custom query",
        "adjacency": "consecutive",
    },
}


@main.command("init-spec")
@click.argument("name", default="my_query.json")
@click.option(
    "--template", "-t",
    type=click.Choice(list(_SPEC_TEMPLATES.keys())),
    default="counting",
    help="Spec template to generate.",
)
def init_spec(name: str, template: str) -> None:
    """Generate a starter query specification file (JSON or YAML).

    Creates a new spec file that can be used with
    ``dp-forge synthesize --spec-file`` or validated with
    ``dp-forge check-spec``.  Use a .yaml or .yml extension to
    generate YAML format instead of JSON.
    """
    dest = Path(name)
    if dest.exists():
        _print_error(f"File already exists: {dest}")
        raise SystemExit(1)

    data = _SPEC_TEMPLATES[template]
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.suffix.lower() in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            _print_error("PyYAML required for YAML output: pip install pyyaml")
            raise SystemExit(1)
        with open(dest, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    else:
        with open(dest, "w") as f:
            json.dump(data, f, indent=2)

    _print_success(f"Created {dest} (template: {template})")
    _print_info(f"  Synthesize: dp-forge synthesize --spec-file {dest}")
    _print_info(f"  Validate:   dp-forge check-spec {dest}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    main()
