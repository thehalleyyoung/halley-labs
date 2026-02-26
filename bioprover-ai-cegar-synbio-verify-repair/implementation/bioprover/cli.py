"""BioProver command-line interface.

Entry point for verification, synthesis, repair, benchmarking, and
model inspection from the terminal.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

__version__ = "0.1.0"

logger = logging.getLogger("bioprover.cli")

# Exit codes
EXIT_VERIFIED = 0
EXIT_FALSIFIED = 1
EXIT_UNKNOWN = 2
EXIT_ERROR = 3


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with all subcommands.

    Returns:
        Configured :class:`argparse.ArgumentParser` ready for
        :meth:`parse_args`.
    """
    parser = argparse.ArgumentParser(
        prog="bioprover",
        description=(
            "BioProver — CEGAR-based verification and repair for "
            "synthetic biology circuits."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Global options
    parser.add_argument(
        "--config", "-c",
        metavar="FILE",
        help="Configuration file (YAML or JSON).",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v=INFO, -vv=DEBUG, -vvv=TRACE).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colour output.",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # -- verify ----------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify",
        help="Verify a biological model against a Bio-STL specification.",
    )
    p_verify.add_argument(
        "--model", "-m", required=True, metavar="FILE",
        help="Path to SBML model file.",
    )
    p_verify.add_argument(
        "--spec", "-s", required=True, metavar="SPEC",
        help="Bio-STL specification (file path or inline string).",
    )
    p_verify.add_argument(
        "--mode", choices=["full", "bounded", "compositional"], default="full",
        help="Verification mode (default: full).",
    )
    p_verify.add_argument(
        "--timeout", "-t", type=int, default=300,
        help="Maximum time in seconds (default: 300).",
    )
    p_verify.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write results to FILE.",
    )
    p_verify.add_argument(
        "--format", "-f",
        choices=["json", "csv", "latex", "html", "text"], default="text",
        help="Output format (default: text).",
    )
    p_verify.set_defaults(func=cmd_verify)

    # -- synthesize ------------------------------------------------------------
    p_synth = subparsers.add_parser(
        "synthesize",
        help="Synthesize parameters satisfying a Bio-STL specification.",
    )
    p_synth.add_argument(
        "--model", "-m", required=True, metavar="FILE",
        help="Path to SBML model file.",
    )
    p_synth.add_argument(
        "--spec", "-s", required=True, metavar="SPEC",
        help="Bio-STL specification (file path or inline string).",
    )
    p_synth.add_argument(
        "--objective",
        choices=["feasibility", "robustness", "minimal"],
        default="feasibility",
        help="Optimization objective (default: feasibility).",
    )
    p_synth.add_argument(
        "--timeout", "-t", type=int, default=600,
        help="Maximum time in seconds (default: 600).",
    )
    p_synth.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write results to FILE.",
    )
    p_synth.set_defaults(func=cmd_synthesize)

    # -- repair ----------------------------------------------------------------
    p_repair = subparsers.add_parser(
        "repair",
        help="Repair model parameters to satisfy a specification.",
    )
    p_repair.add_argument(
        "--model", "-m", required=True, metavar="FILE",
        help="Path to SBML model file.",
    )
    p_repair.add_argument(
        "--spec", "-s", required=True, metavar="SPEC",
        help="Bio-STL specification (file path or inline string).",
    )
    p_repair.add_argument(
        "--budget", "-b", type=float, default=0.5,
        help="Perturbation budget as a fraction (default: 0.5).",
    )
    p_repair.add_argument(
        "--timeout", "-t", type=int, default=600,
        help="Maximum time in seconds (default: 600).",
    )
    p_repair.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write results to FILE.",
    )
    p_repair.set_defaults(func=cmd_repair)

    # -- benchmark -------------------------------------------------------------
    p_bench = subparsers.add_parser(
        "benchmark",
        help="Run benchmark suite and compare against baselines.",
    )
    p_bench.add_argument(
        "--suite",
        choices=["toggle_switch", "repressilator", "full"],
        default="full",
        help="Benchmark suite to run (default: full).",
    )
    p_bench.add_argument(
        "--baselines",
        nargs="+",
        choices=["bioprover", "dreach", "flow_star", "spaceex"],
        help="Baselines to compare against.",
    )
    p_bench.add_argument(
        "--output", "-o", metavar="FILE",
        help="Write benchmark results to FILE.",
    )
    p_bench.add_argument(
        "--format", "-f",
        choices=["json", "csv", "latex", "html", "text"], default="csv",
        help="Output format (default: csv).",
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # -- info ------------------------------------------------------------------
    p_info = subparsers.add_parser(
        "info",
        help="Display model information and optional specification summary.",
    )
    p_info.add_argument(
        "--model", "-m", required=True, metavar="FILE",
        help="Path to SBML model file.",
    )
    p_info.add_argument(
        "--spec", "-s", metavar="SPEC",
        help="Optional Bio-STL specification for summary.",
    )
    p_info.set_defaults(func=cmd_info)

    return parser


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def load_config(path: str) -> Dict[str, Any]:
    """Load a configuration file (JSON or YAML).

    Args:
        path: Filesystem path to the configuration file.

    Returns:
        Parsed configuration as a dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
        ValueError: If the file format is unsupported or malformed.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # type: ignore[import-untyped]
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required to load YAML configuration files. "
                "Install it with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    elif suffix == ".json":
        data = json.loads(text)
    else:
        raise ValueError(
            f"Unsupported configuration format '{suffix}'. "
            "Use .json or .yaml/.yml."
        )

    if not isinstance(data, dict):
        raise ValueError("Configuration file must contain a top-level mapping.")
    return data


def load_model(path: str) -> Any:
    """Import an SBML model from *path*.

    Returns:
        A :class:`~bioprover.models.bio_model.BioModel` instance.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    from bioprover.models.sbml_import import SBMLImporter  # noqa: WPS433

    importer = SBMLImporter()
    return importer.import_model(path)


def load_spec(spec_arg: str) -> Any:
    """Load a Bio-STL specification from a file or inline string.

    If *spec_arg* points to an existing file the contents are read first;
    otherwise *spec_arg* is treated as an inline specification string.

    Returns:
        The parsed specification object produced by
        :class:`~bioprover.temporal.bio_stl_parser.BioSTLParser`.
    """
    from bioprover.temporal.bio_stl_parser import BioSTLParser  # noqa: WPS433

    spec_path = Path(spec_arg)
    if spec_path.is_file():
        spec_text = spec_path.read_text(encoding="utf-8")
    else:
        spec_text = spec_arg

    parser = BioSTLParser()
    return parser.parse(spec_text)


def format_result_text(result: Any) -> str:
    """Format a :class:`VerificationResult` as human-readable text.

    Args:
        result: A :class:`~bioprover.cegar.cegar_engine.VerificationResult`.

    Returns:
        A multi-line string suitable for terminal display.
    """
    lines = [
        "=" * 60,
        "  BioProver Verification Result",
        "=" * 60,
        f"  Status          : {result.status.name}",
        f"  Iterations      : {result.iterations}",
        f"  Elapsed time    : {result.elapsed_time:.2f}s",
    ]
    if hasattr(result, "counterexample") and result.counterexample is not None:
        lines.append(f"  Counterexample  : yes ({len(result.counterexample)} states)")
    else:
        lines.append("  Counterexample  : none")
    if hasattr(result, "message") and result.message:
        lines.append(f"  Message         : {result.message}")
    lines.append("=" * 60)
    return "\n".join(lines)


def _setup_verbosity(verbose: int, quiet: bool) -> None:
    """Configure the root logger level from CLI flags."""
    if quiet:
        level = logging.WARNING
    elif verbose >= 3:
        level = logging.DEBUG - 5  # TRACE-like
    elif verbose == 2:
        level = logging.DEBUG
    elif verbose == 1:
        level = logging.INFO
    else:
        level = logging.WARNING

    try:
        from bioprover.visualization.progress import setup_logging  # noqa: WPS433
        setup_logging(level=level)
    except ImportError:
        logging.basicConfig(level=level, format="%(levelname)s: %(message)s")


def _export_result(
    result: Any,
    output_path: str,
    fmt: str,
) -> None:
    """Write *result* to *output_path* using the requested format."""
    from bioprover.visualization.result_export import (  # noqa: WPS433
        ExportFormat,
        ResultExporter,
    )

    format_map: Dict[str, ExportFormat] = {
        "json": ExportFormat.JSON,
        "csv": ExportFormat.CSV,
        "latex": ExportFormat.LATEX,
        "html": ExportFormat.HTML,
        "text": ExportFormat.TEXT,
    }
    export_fmt = format_map.get(fmt, ExportFormat.TEXT)
    exporter = ResultExporter(export_fmt)
    exporter.export(result, output_path)
    logger.info("Results written to %s", output_path)


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_verify(args: argparse.Namespace) -> int:
    """Run CEGAR-based model verification.

    Returns:
        Exit code: 0 verified, 1 falsified, 2 unknown, 3 error.
    """
    try:
        from bioprover.cegar.cegar_engine import (  # noqa: WPS433
            CEGARConfig,
            CEGAREngine,
            VerificationStatus,
        )
        from bioprover.visualization.progress import (  # noqa: WPS433
            ProgressConfig,
            ProgressReporter,
        )
    except ImportError as exc:
        print(f"Error: required BioProver modules not found: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        model = load_model(args.model)
        spec = load_spec(args.spec)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    config = CEGARConfig(
        mode=args.mode,
        timeout=args.timeout,
    )

    reporter: Optional[ProgressReporter] = None
    if not args.quiet:
        reporter = ProgressReporter(ProgressConfig(colour=not args.no_color))

    engine = CEGAREngine(config)
    try:
        result = engine.verify(model, spec, progress=reporter)
    except TimeoutError:
        print("Error: verification timed out.", file=sys.stderr)
        return EXIT_UNKNOWN
    except KeyboardInterrupt:
        print("\nVerification interrupted by user.", file=sys.stderr)
        return EXIT_ERROR

    # Display result
    print(format_result_text(result))

    # Export if requested
    if args.output:
        try:
            _export_result(result, args.output, args.format)
        except (ImportError, OSError) as exc:
            print(f"Error writing output: {exc}", file=sys.stderr)
            return EXIT_ERROR

    # Map status to exit code
    status_map = {
        VerificationStatus.VERIFIED: EXIT_VERIFIED,
        VerificationStatus.FALSIFIED: EXIT_FALSIFIED,
        VerificationStatus.UNKNOWN: EXIT_UNKNOWN,
    }
    return status_map.get(result.status, EXIT_UNKNOWN)


def cmd_synthesize(args: argparse.Namespace) -> int:
    """Run parameter synthesis for a biological model.

    Returns:
        Exit code: 0 success, 3 error.
    """
    try:
        from bioprover.repair.parameter_synthesis import (  # noqa: WPS433
            ParameterSynthesizer,
            SynthesisConfig,
            SynthesisMode,
        )
    except ImportError as exc:
        print(f"Error: required BioProver modules not found: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        model = load_model(args.model)
        spec = load_spec(args.spec)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    mode_map: Dict[str, SynthesisMode] = {
        "feasibility": SynthesisMode.FEASIBILITY,
        "robustness": SynthesisMode.ROBUSTNESS,
        "minimal": SynthesisMode.MINIMAL,
    }
    synthesis_mode = mode_map[args.objective]

    config = SynthesisConfig(
        mode=synthesis_mode,
        timeout=args.timeout,
    )
    synthesizer = ParameterSynthesizer(config)

    try:
        result = synthesizer.synthesize(model, spec)
    except TimeoutError:
        print("Error: synthesis timed out.", file=sys.stderr)
        return EXIT_UNKNOWN
    except KeyboardInterrupt:
        print("\nSynthesis interrupted by user.", file=sys.stderr)
        return EXIT_ERROR

    # Print summary
    print("=" * 60)
    print("  BioProver Parameter Synthesis Result")
    print("=" * 60)
    print(f"  Feasible        : {result.feasible}")
    print(f"  Elapsed time    : {result.elapsed_time:.2f}s")
    if hasattr(result, "parameters") and result.parameters:
        print("  Parameters      :")
        for name, value in result.parameters.items():
            print(f"    {name:20s} = {value}")
    print("=" * 60)

    if args.output:
        try:
            _export_result(result, args.output, "json")
        except (ImportError, OSError) as exc:
            print(f"Error writing output: {exc}", file=sys.stderr)
            return EXIT_ERROR

    return EXIT_VERIFIED if result.feasible else EXIT_FALSIFIED


def cmd_repair(args: argparse.Namespace) -> int:
    """Run parameter repair to satisfy a specification.

    Returns:
        Exit code: 0 repaired, 1 not repairable, 3 error.
    """
    try:
        from bioprover.repair.parameter_synthesis import (  # noqa: WPS433
            ParameterSynthesizer,
            SynthesisConfig,
            SynthesisMode,
        )
        from bioprover.repair.repair_report import RepairReport  # noqa: WPS433
    except ImportError as exc:
        print(f"Error: required BioProver modules not found: {exc}", file=sys.stderr)
        return EXIT_ERROR

    try:
        model = load_model(args.model)
        spec = load_spec(args.spec)
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    config = SynthesisConfig(
        mode=SynthesisMode.MINIMAL,
        timeout=args.timeout,
    )
    synthesizer = ParameterSynthesizer(config)

    try:
        result = synthesizer.synthesize(
            model,
            spec,
            budget=args.budget,
        )
    except TimeoutError:
        print("Error: repair timed out.", file=sys.stderr)
        return EXIT_UNKNOWN
    except KeyboardInterrupt:
        print("\nRepair interrupted by user.", file=sys.stderr)
        return EXIT_ERROR

    report = RepairReport(result, budget=args.budget)

    print("=" * 60)
    print("  BioProver Repair Report")
    print("=" * 60)
    print(f"  Repaired        : {report.success}")
    print(f"  Budget used     : {report.budget_used:.2%}")
    print(f"  Elapsed time    : {result.elapsed_time:.2f}s")
    if report.success and hasattr(report, "changes") and report.changes:
        print("  Parameter changes:")
        for change in report.changes:
            print(f"    {change.name:20s}: {change.old_value} -> {change.new_value}")
    print("=" * 60)

    if args.output:
        try:
            _export_result(report, args.output, "json")
        except (ImportError, OSError) as exc:
            print(f"Error writing output: {exc}", file=sys.stderr)
            return EXIT_ERROR

    return EXIT_VERIFIED if report.success else EXIT_FALSIFIED


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run a benchmark suite and compare tools.

    Returns:
        Exit code: 0 success, 3 error.
    """
    try:
        from bioprover.cegar.cegar_engine import (  # noqa: WPS433
            CEGARConfig,
            CEGAREngine,
        )
        from bioprover.visualization.result_export import (  # noqa: WPS433
            ExportFormat,
            ResultExporter,
        )
    except ImportError as exc:
        print(f"Error: required BioProver modules not found: {exc}", file=sys.stderr)
        return EXIT_ERROR

    baselines = args.baselines or ["bioprover"]
    suite = args.suite

    print(f"Running benchmark suite '{suite}' with baselines: {', '.join(baselines)}")
    print("-" * 60)

    # Benchmark suite definitions — model file and spec pairs
    suites: Dict[str, list[Dict[str, str]]] = {
        "toggle_switch": [
            {"model": "benchmarks/toggle_switch.xml", "spec": "benchmarks/toggle_switch.stl"},
        ],
        "repressilator": [
            {"model": "benchmarks/repressilator.xml", "spec": "benchmarks/repressilator.stl"},
        ],
        "full": [
            {"model": "benchmarks/toggle_switch.xml", "spec": "benchmarks/toggle_switch.stl"},
            {"model": "benchmarks/repressilator.xml", "spec": "benchmarks/repressilator.stl"},
        ],
    }

    bench_cases = suites.get(suite, suites["full"])
    results: list[Dict[str, Any]] = []

    for case in bench_cases:
        model_path = case["model"]
        spec_path = case["spec"]
        case_name = Path(model_path).stem

        for baseline in baselines:
            print(f"  {case_name} / {baseline} ... ", end="", flush=True)
            start = time.monotonic()

            if baseline == "bioprover":
                try:
                    model = load_model(model_path)
                    spec = load_spec(spec_path)
                    engine = CEGAREngine(CEGARConfig(timeout=300))
                    vr = engine.verify(model, spec)
                    elapsed = time.monotonic() - start
                    results.append({
                        "case": case_name,
                        "tool": baseline,
                        "status": vr.status.name,
                        "time": round(elapsed, 3),
                        "iterations": vr.iterations,
                    })
                    print(f"{vr.status.name} ({elapsed:.2f}s)")
                except Exception as exc:  # noqa: BLE001
                    elapsed = time.monotonic() - start
                    results.append({
                        "case": case_name,
                        "tool": baseline,
                        "status": "ERROR",
                        "time": round(elapsed, 3),
                        "iterations": 0,
                    })
                    print(f"ERROR ({exc})")
            else:
                # External baselines are not executed; placeholder entries.
                elapsed = time.monotonic() - start
                results.append({
                    "case": case_name,
                    "tool": baseline,
                    "status": "N/A",
                    "time": 0.0,
                    "iterations": 0,
                })
                print("N/A (external tool not integrated)")

    print("-" * 60)
    print(f"Completed {len(results)} benchmark runs.")

    if args.output:
        try:
            fmt = getattr(args, "format", "csv")
            _export_result(results, args.output, fmt)
        except (ImportError, OSError) as exc:
            print(f"Error writing output: {exc}", file=sys.stderr)
            return EXIT_ERROR

    return EXIT_VERIFIED


def cmd_info(args: argparse.Namespace) -> int:
    """Display model information and optional specification summary.

    Returns:
        Exit code: 0 success, 3 error.
    """
    try:
        model = load_model(args.model)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR

    print("=" * 60)
    print("  Model Information")
    print("=" * 60)
    print(f"  File            : {args.model}")
    print(f"  Name            : {getattr(model, 'name', 'N/A')}")

    species = getattr(model, "species", [])
    print(f"  Species         : {len(species)}")
    for sp in species:
        name = getattr(sp, "name", str(sp))
        print(f"    - {name}")

    parameters = getattr(model, "parameters", [])
    print(f"  Parameters      : {len(parameters)}")
    for param in parameters:
        p_name = getattr(param, "name", str(param))
        p_value = getattr(param, "value", "?")
        print(f"    - {p_name} = {p_value}")

    reactions = getattr(model, "reactions", [])
    print(f"  Reactions       : {len(reactions)}")
    for rxn in reactions:
        r_name = getattr(rxn, "name", str(rxn))
        print(f"    - {r_name}")

    if args.spec:
        try:
            spec = load_spec(args.spec)
            print()
            print("  Specification")
            print("  " + "-" * 40)
            print(f"  Source          : {args.spec}")
            print(f"  Parsed          : {spec}")
        except (ValueError, ImportError) as exc:
            print(f"  Spec parse error: {exc}", file=sys.stderr)

    print("=" * 60)
    return EXIT_VERIFIED


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[list[str]] = None) -> int:
    """BioProver CLI entry point.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns:
        Process exit code.
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    # Apply config file overrides if provided
    if args.config:
        try:
            config = load_config(args.config)
            for key, value in config.items():
                if not hasattr(args, key) or getattr(args, key) is None:
                    setattr(args, key, value)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            print(f"Error loading config: {exc}", file=sys.stderr)
            return EXIT_ERROR

    _setup_verbosity(args.verbose, args.quiet)

    if args.command is None:
        parser.print_help()
        return EXIT_ERROR

    handler = args.func
    try:
        return handler(args)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_ERROR
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unhandled exception", exc_info=True)
        print(f"Error: {exc}", file=sys.stderr)
        return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
