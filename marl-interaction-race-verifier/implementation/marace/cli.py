"""MARACE CLI: Command-line interface for the Multi-Agent Race Condition Verifier.

Provides subcommands for verification, trace analysis, benchmarking,
reporting, policy inspection, replay, and specification validation.

Entry point is registered as ``marace = "marace.cli:main"`` in pyproject.toml.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import textwrap
import time
from pathlib import Path
from typing import Any, Sequence

# ---------------------------------------------------------------------------
# Package metadata (imported lazily to keep CLI startup fast)
# ---------------------------------------------------------------------------

_VERSION: str | None = None


def _get_version() -> str:
    """Return the package version string."""
    global _VERSION  # noqa: PLW0603
    if _VERSION is None:
        try:
            from marace import __version__

            _VERSION = __version__
        except ImportError:
            _VERSION = "0.1.0"
    return _VERSION


# ---------------------------------------------------------------------------
# ColorOutput – ANSI terminal colours
# ---------------------------------------------------------------------------


class ColorOutput:
    """Pretty-print messages with ANSI escape codes.

    Colours are automatically disabled when *stdout* is not a TTY (e.g. when
    output is piped to a file or another process).
    """

    RESET = "\033[0m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    BOLD = "\033[1m"

    def __init__(self, *, force_color: bool = False) -> None:
        self._enabled = force_color or (
            hasattr(sys.stdout, "isatty") and sys.stdout.isatty()
        )

    def _wrap(self, code: str, msg: str) -> str:
        if self._enabled:
            return f"{code}{msg}{self.RESET}"
        return msg

    def success(self, msg: str) -> str:
        """Return *msg* styled as a success (green)."""
        return self._wrap(self.GREEN, f"✓ {msg}")

    def error(self, msg: str) -> str:
        """Return *msg* styled as an error (red, bold)."""
        return self._wrap(f"{self.BOLD}{self.RED}", f"✗ {msg}")

    def warning(self, msg: str) -> str:
        """Return *msg* styled as a warning (yellow)."""
        return self._wrap(self.YELLOW, f"⚠ {msg}")

    def info(self, msg: str) -> str:
        """Return *msg* styled as informational (cyan)."""
        return self._wrap(self.CYAN, f"ℹ {msg}")

    def header(self, msg: str) -> str:
        """Return *msg* styled as a section header (bold magenta)."""
        return self._wrap(f"{self.BOLD}{self.MAGENTA}", msg)

    def bold(self, msg: str) -> str:
        """Return *msg* in bold."""
        return self._wrap(self.BOLD, msg)


# Shared instance
_color = ColorOutput()

# ---------------------------------------------------------------------------
# ConfigLoader – YAML / JSON configuration files
# ---------------------------------------------------------------------------


class ConfigLoader:
    """Load, merge, and validate configuration from YAML or JSON files."""

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        """Load a configuration file (YAML or JSON) and return a dict.

        Raises ``FileNotFoundError`` if *path* does not exist and
        ``ValueError`` if the format is unsupported.
        """
        filepath = Path(path)
        if not filepath.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = filepath.suffix.lower()
        text = filepath.read_text(encoding="utf-8")

        if suffix in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError as exc:
                raise ImportError(
                    "PyYAML is required for YAML config files. "
                    "Install it with: pip install pyyaml"
                ) from exc
            return dict(yaml.safe_load(text) or {})

        if suffix == ".json":
            return dict(json.loads(text))

        raise ValueError(
            f"Unsupported configuration format '{suffix}'. Use .yaml, .yml, or .json"
        )

    @staticmethod
    def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
        """Recursively merge *override* into *base* (non-destructive).

        Returns a new dict.  Nested dicts are merged; all other values in
        *override* replace those in *base*.
        """
        merged: dict[str, Any] = dict(base)
        for key, value in override.items():
            if (
                key in merged
                and isinstance(merged[key], dict)
                and isinstance(value, dict)
            ):
                merged[key] = ConfigLoader.merge(merged[key], value)
            else:
                merged[key] = value
        return merged

    @staticmethod
    def validate(config: dict[str, Any], schema: str) -> list[str]:
        """Validate *config* against a named schema.

        Returns a (possibly empty) list of human-readable error strings.
        Currently supports lightweight built-in checks; a JSON-Schema
        backend will be added when the ``jsonschema`` package is available.
        """
        errors: list[str] = []

        if schema == "environment":
            for required in ("observation_space", "action_space", "num_agents"):
                if required not in config:
                    errors.append(f"Missing required field '{required}'")

        elif schema == "specification":
            if "properties" not in config and "safety" not in config:
                errors.append(
                    "Specification must define at least 'properties' or 'safety'"
                )

        elif schema == "benchmark":
            if "suites" not in config and "scenarios" not in config:
                errors.append(
                    "Benchmark config must define 'suites' or 'scenarios'"
                )

        else:
            errors.append(f"Unknown schema '{schema}'")

        return errors


# ---------------------------------------------------------------------------
# ProgressBar – minimal terminal progress indicator
# ---------------------------------------------------------------------------


class ProgressBar:
    """Simple terminal progress bar for long-running operations.

    Example::

        bar = ProgressBar(total=100, desc="Verifying")
        for i in range(100):
            do_work()
            bar.update()
        bar.finish()
    """

    def __init__(self, total: int, width: int = 40, desc: str = "") -> None:
        self.total = max(total, 1)
        self.width = width
        self.desc = desc
        self.current = 0
        self._start_time = time.monotonic()

    def update(self, n: int = 1) -> None:
        """Advance the progress bar by *n* steps and redraw."""
        self.current = min(self.current + n, self.total)
        sys.stderr.write(f"\r{self}")
        sys.stderr.flush()

    def finish(self) -> None:
        """Mark the bar as complete and print a newline."""
        self.current = self.total
        sys.stderr.write(f"\r{self}\n")
        sys.stderr.flush()

    def __str__(self) -> str:
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        pct = frac * 100
        elapsed = time.monotonic() - self._start_time
        prefix = f"{self.desc}: " if self.desc else ""
        return f"{prefix}|{bar}| {pct:5.1f}% [{self.current}/{self.total}] {elapsed:.1f}s"


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def _setup_logging(verbose: bool, quiet: bool) -> None:
    """Configure the root logger for the CLI session.

    Parameters
    ----------
    verbose:
        If *True*, set level to ``DEBUG``.
    quiet:
        If *True*, set level to ``WARNING`` (overrides *verbose*).
    """
    if quiet:
        level = logging.WARNING
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )


# ---------------------------------------------------------------------------
# Banner
# ---------------------------------------------------------------------------


def _print_banner() -> None:
    """Print the MARACE banner with version information."""
    banner = textwrap.dedent(f"""\
        {_color.header("╔══════════════════════════════════════════════════╗")}
        {_color.header("║")}  {_color.bold("MARACE")} — Multi-Agent Race Condition Verifier   {_color.header("║")}
        {_color.header("║")}  Version {_get_version():<40s} {_color.header("║")}
        {_color.header("╚══════════════════════════════════════════════════╝")}
    """)
    print(banner, file=sys.stderr)


# ---------------------------------------------------------------------------
# Helper: safe output writing
# ---------------------------------------------------------------------------


def _write_output(data: str, output_path: str | None, label: str = "results") -> None:
    """Write *data* to *output_path* (or stdout if ``None``)."""
    if output_path is None:
        print(data)
    else:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(data, encoding="utf-8")
        print(_color.success(f"{label} written to {out}"), file=sys.stderr)


def _ensure_output_dir(path: str | None) -> Path | None:
    """Create the output directory if it does not exist.  Returns the Path."""
    if path is None:
        return None
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def cmd_verify(args: argparse.Namespace) -> int:
    """Run the full MARACE verification pipeline.

    Loads the environment configuration, policy files, and safety
    specification, then invokes each pipeline stage (abstraction,
    happens-before, search, sampling) and writes the final report.
    """
    _print_banner()
    logger = logging.getLogger("marace.verify")

    # Validate inputs -------------------------------------------------------
    if not Path(args.env).exists():
        print(_color.error(f"Environment config not found: {args.env}"), file=sys.stderr)
        return 1

    for pf in args.policies:
        if not Path(pf).exists():
            print(_color.error(f"Policy file not found: {pf}"), file=sys.stderr)
            return 1

    if args.spec and not Path(args.spec).exists():
        print(_color.error(f"Specification file not found: {args.spec}"), file=sys.stderr)
        return 1

    # Load configuration ----------------------------------------------------
    loader = ConfigLoader()
    try:
        env_config = loader.load(args.env)
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(_color.error(str(exc)), file=sys.stderr)
        return 1

    validation_errors = loader.validate(env_config, "environment")
    if validation_errors:
        for err in validation_errors:
            print(_color.warning(err), file=sys.stderr)

    spec_config: dict[str, Any] = {}
    if args.spec:
        try:
            spec_config = loader.load(args.spec)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            print(_color.error(str(exc)), file=sys.stderr)
            return 1

    output_dir = _ensure_output_dir(args.output)
    logger.info("Environment: %s (%d keys)", args.env, len(env_config))
    logger.info("Policies: %s", ", ".join(args.policies))
    logger.info("Parallel: %s, Timeout: %ss", args.parallel, args.timeout)

    # Attempt to import the pipeline ----------------------------------------
    try:
        from marace import pipeline  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "The verification pipeline module is not yet available.\n"
                "  Ensure 'marace.pipeline' is implemented and importable.\n"
                "  Required sub-modules: abstract, hb, search, sampling."
            ),
            file=sys.stderr,
        )
        # Simulate progress for UI demonstration
        bar = ProgressBar(total=5, desc="Pipeline stages")
        stages = [
            "Abstract interpretation",
            "Happens-before construction",
            "Adversarial search",
            "Importance sampling",
            "Report generation",
        ]
        for stage in stages:
            logger.info("Stage: %s (not yet implemented)", stage)
            time.sleep(0.1)
            bar.update()
        bar.finish()

        summary = {
            "status": "unavailable",
            "message": "Pipeline modules not yet implemented",
            "env_config": args.env,
            "policies": args.policies,
            "spec": args.spec,
        }
        output_file = str(output_dir / f"verify_result.{args.format}") if output_dir else None
        _write_output(json.dumps(summary, indent=2), output_file, label="Verification stub")
        return 0

    # Full pipeline execution (when modules are available) ------------------
    logger.info("Starting verification pipeline")
    bar = ProgressBar(total=5, desc="Verifying")

    try:
        result = pipeline.run(
            env_config=env_config,
            policy_paths=[str(p) for p in args.policies],
            spec=spec_config,
            parallel=args.parallel,
            timeout=args.timeout,
            checkpoint_dir=args.checkpoint_dir,
        )
        bar.update(5)
        bar.finish()
    except Exception as exc:
        bar.finish()
        logger.exception("Verification failed")
        print(_color.error(f"Verification failed: {exc}"), file=sys.stderr)
        return 1

    output_file = str(output_dir / f"verify_result.{args.format}") if output_dir else None
    _write_output(json.dumps(result, indent=2, default=str), output_file, label="Verification")
    print(_color.success("Verification complete"), file=sys.stderr)
    return 0


def cmd_analyze_trace(args: argparse.Namespace) -> int:
    """Analyze a recorded multi-agent execution trace for race conditions."""
    _print_banner()
    logger = logging.getLogger("marace.analyze_trace")

    trace_path = Path(args.trace_file)
    if not trace_path.exists():
        print(_color.error(f"Trace file not found: {trace_path}"), file=sys.stderr)
        return 1

    logger.info("Analyzing trace: %s", trace_path)

    # Load specification if provided
    spec_config: dict[str, Any] = {}
    if args.spec:
        try:
            spec_config = ConfigLoader.load(args.spec)
        except (FileNotFoundError, ValueError, ImportError) as exc:
            print(_color.error(str(exc)), file=sys.stderr)
            return 1

    try:
        from marace import trace as trace_mod  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "Trace analysis module not yet available.\n"
                "  Ensure 'marace.trace' provides an 'analyze' function."
            ),
            file=sys.stderr,
        )
        # Stub: read trace and report basic statistics
        try:
            raw = trace_path.read_text(encoding="utf-8")
            trace_data = json.loads(raw)
            num_steps = len(trace_data) if isinstance(trace_data, list) else 1
        except (json.JSONDecodeError, UnicodeDecodeError):
            num_steps = 0

        summary = {
            "trace_file": str(trace_path),
            "steps_loaded": num_steps,
            "status": "stub",
            "message": "Full analysis requires marace.trace module",
        }
        _write_output(json.dumps(summary, indent=2), args.output, label="Trace analysis stub")
        return 0

    bar = ProgressBar(total=3, desc="Analyzing")
    try:
        loaded = trace_mod.load(str(trace_path))
        bar.update()
        analysis = trace_mod.analyze(loaded, spec=spec_config)
        bar.update()
        report = trace_mod.format_report(analysis, fmt=args.format)
        bar.update()
        bar.finish()
    except Exception as exc:
        bar.finish()
        logger.exception("Trace analysis failed")
        print(_color.error(f"Analysis failed: {exc}"), file=sys.stderr)
        return 1

    _write_output(report, args.output, label="Trace analysis")
    print(_color.success("Trace analysis complete"), file=sys.stderr)
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the MARACE benchmark suite."""
    _print_banner()
    logger = logging.getLogger("marace.benchmark")
    logger.info("Suite: %s, Agents: %s, Timeout: %ss", args.suite, args.num_agents, args.timeout)

    output_dir = _ensure_output_dir(args.output)

    try:
        from marace import evaluation  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "Benchmark module not fully available.\n"
                "  Ensure 'marace.evaluation' provides benchmark utilities."
            ),
            file=sys.stderr,
        )
        suites = ["default"] if args.suite == "default" else ["default", "scalability"]
        bar = ProgressBar(total=len(suites), desc="Benchmarks")
        results: dict[str, Any] = {"suites": {}}
        for suite_name in suites:
            logger.info("Running suite: %s (stub)", suite_name)
            results["suites"][suite_name] = {
                "status": "stub",
                "scenarios": 0,
                "message": "Benchmark logic not yet implemented",
            }
            time.sleep(0.05)
            bar.update()
        bar.finish()

        if args.compare_baselines:
            results["baselines"] = {"note": "Baseline comparison not yet available"}

        output_file = str(output_dir / "benchmark_results.json") if output_dir else None
        _write_output(json.dumps(results, indent=2), output_file, label="Benchmark results")
        return 0

    bar = ProgressBar(total=1, desc="Benchmarks")
    try:
        results = evaluation.run_benchmarks(
            suite=args.suite,
            num_agents=args.num_agents,
            timeout=args.timeout,
            compare_baselines=args.compare_baselines,
        )
        bar.update()
        bar.finish()
    except Exception as exc:
        bar.finish()
        logger.exception("Benchmark failed")
        print(_color.error(f"Benchmark failed: {exc}"), file=sys.stderr)
        return 1

    output_file = str(output_dir / "benchmark_results.json") if output_dir else None
    _write_output(json.dumps(results, indent=2, default=str), output_file, label="Benchmark")
    print(_color.success("Benchmark complete"), file=sys.stderr)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Generate a formatted report from saved verification results."""
    _print_banner()
    logger = logging.getLogger("marace.report")

    results_path = Path(args.results_file)
    if not results_path.exists():
        print(_color.error(f"Results file not found: {results_path}"), file=sys.stderr)
        return 1

    try:
        raw = results_path.read_text(encoding="utf-8")
        results_data = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        print(_color.error(f"Failed to parse results file: {exc}"), file=sys.stderr)
        return 1

    logger.info("Loaded results from %s", results_path)

    try:
        from marace import reporting  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "Reporting module not fully available.\n"
                "  Ensure 'marace.reporting' is implemented."
            ),
            file=sys.stderr,
        )
        # Minimal stub report
        bar = ProgressBar(total=2, desc="Report")
        report_lines = [
            f"MARACE Report — generated {time.strftime('%Y-%m-%d %H:%M:%S')}",
            "=" * 60,
            "",
            f"Source: {results_path}",
            f"Format: {args.format}",
            f"Include certificates: {args.include_certificates}",
            "",
            json.dumps(results_data, indent=2, default=str),
        ]
        bar.update(2)
        bar.finish()
        report_text = "\n".join(report_lines)
        _write_output(report_text, args.output, label="Report")
        return 0

    bar = ProgressBar(total=2, desc="Report")
    try:
        report_text = reporting.generate(
            results_data,
            fmt=args.format,
            include_certificates=args.include_certificates,
        )
        bar.update(2)
        bar.finish()
    except Exception as exc:
        bar.finish()
        logger.exception("Report generation failed")
        print(_color.error(f"Report generation failed: {exc}"), file=sys.stderr)
        return 1

    _write_output(report_text, args.output, label="Report")
    print(_color.success("Report generated"), file=sys.stderr)
    return 0


def cmd_inspect_policy(args: argparse.Namespace) -> int:
    """Inspect and analyze a MARL policy file."""
    _print_banner()
    logger = logging.getLogger("marace.inspect_policy")

    policy_path = Path(args.policy_file)
    if not policy_path.exists():
        print(_color.error(f"Policy file not found: {policy_path}"), file=sys.stderr)
        return 1

    logger.info("Inspecting policy: %s (format=%s)", policy_path, args.format)

    try:
        from marace import policy as policy_mod  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "Policy inspection module not fully available.\n"
                "  Ensure 'marace.policy' provides 'load' and 'inspect' functions."
            ),
            file=sys.stderr,
        )
        # Provide file-level info as a stub
        stat = policy_path.stat()
        info = {
            "file": str(policy_path),
            "size_bytes": stat.st_size,
            "format": args.format,
            "detail_level": "detailed" if args.detailed else "summary",
            "status": "stub",
            "message": "Full inspection requires marace.policy module",
        }
        print(json.dumps(info, indent=2))
        return 0

    try:
        loaded = policy_mod.load(str(policy_path), fmt=args.format)
        detail = "detailed" if args.detailed else "summary"
        report = policy_mod.inspect(loaded, detail=detail)
    except Exception as exc:
        logger.exception("Policy inspection failed")
        print(_color.error(f"Inspection failed: {exc}"), file=sys.stderr)
        return 1

    print(report)
    print(_color.success("Policy inspection complete"), file=sys.stderr)
    return 0


def cmd_replay(args: argparse.Namespace) -> int:
    """Replay an adversarial trace interactively."""
    _print_banner()
    logger = logging.getLogger("marace.replay")

    replay_path = Path(args.replay_file)
    if not replay_path.exists():
        print(_color.error(f"Replay file not found: {replay_path}"), file=sys.stderr)
        return 1

    mode = "step" if args.step else "continuous"
    logger.info("Replaying %s (mode=%s, speed=%.1fx)", replay_path, mode, args.speed)

    try:
        from marace import trace as trace_mod  # type: ignore[attr-defined]
    except (ImportError, AttributeError):
        print(
            _color.warning(
                "Replay module not fully available.\n"
                "  Ensure 'marace.trace' provides replay capabilities."
            ),
            file=sys.stderr,
        )
        # Stub: show trace metadata
        try:
            raw = replay_path.read_text(encoding="utf-8")
            trace_data = json.loads(raw)
            steps = trace_data if isinstance(trace_data, list) else [trace_data]
        except (json.JSONDecodeError, UnicodeDecodeError):
            print(_color.error("Could not parse replay file as JSON"), file=sys.stderr)
            return 1

        bar = ProgressBar(total=len(steps), desc="Replay")
        for i, step in enumerate(steps):
            label = f"Step {i}"
            if args.highlight_critical and isinstance(step, dict) and step.get("critical"):
                label = _color.warning(f"Step {i} [CRITICAL]")
            print(f"  {label}: {json.dumps(step, default=str)[:120]}")
            bar.update()
            if mode == "continuous":
                time.sleep(0.05 / max(args.speed, 0.01))
        bar.finish()
        return 0

    try:
        loaded = trace_mod.load(str(replay_path))
        trace_mod.replay(
            loaded,
            step_mode=args.step,
            speed=args.speed,
            highlight_critical=args.highlight_critical,
        )
    except Exception as exc:
        logger.exception("Replay failed")
        print(_color.error(f"Replay failed: {exc}"), file=sys.stderr)
        return 1

    print(_color.success("Replay finished"), file=sys.stderr)
    return 0


def cmd_check_spec(args: argparse.Namespace) -> int:
    """Validate a specification file for syntactic and semantic correctness."""
    _print_banner()
    logger = logging.getLogger("marace.check_spec")

    spec_path = Path(args.spec_file)
    if not spec_path.exists():
        print(_color.error(f"Specification file not found: {spec_path}"), file=sys.stderr)
        return 1

    logger.info("Checking specification: %s", spec_path)

    loader = ConfigLoader()
    try:
        spec_config = loader.load(str(spec_path))
    except (FileNotFoundError, ValueError, ImportError) as exc:
        print(_color.error(f"Failed to load spec: {exc}"), file=sys.stderr)
        return 1

    errors = loader.validate(spec_config, "specification")

    # Optionally check compatibility with an environment config
    if args.env:
        if not Path(args.env).exists():
            print(_color.error(f"Environment config not found: {args.env}"), file=sys.stderr)
            return 1
        try:
            env_config = loader.load(args.env)
            env_errors = loader.validate(env_config, "environment")
            if env_errors:
                for e in env_errors:
                    errors.append(f"Environment: {e}")
        except (FileNotFoundError, ValueError, ImportError) as exc:
            errors.append(f"Could not load environment config: {exc}")

    # Also try the spec module if available
    try:
        from marace import spec as spec_mod  # type: ignore[attr-defined]

        if hasattr(spec_mod, "validate"):
            module_errors = spec_mod.validate(spec_config)
            errors.extend(module_errors)
    except (ImportError, AttributeError):
        logger.debug("marace.spec module not available; skipping deep validation")

    if errors:
        print(_color.error(f"Specification has {len(errors)} error(s):"), file=sys.stderr)
        for err in errors:
            print(f"  • {err}", file=sys.stderr)
        return 1

    print(_color.success(f"Specification '{spec_path.name}' is valid"), file=sys.stderr)
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

_OUTPUT_FORMATS = ("text", "json", "html", "latex")


def _build_parser() -> argparse.ArgumentParser:
    """Build and return the top-level argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="marace",
        description="MARACE — Multi-Agent Race Condition Verifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              marace verify --env env.yaml --policies p1.onnx p2.onnx --spec safety.yaml
              marace analyze-trace trace.json --spec safety.yaml -o report.txt
              marace benchmark --suite scalability --num-agents 8
              marace report results.json --format latex -o report.tex
              marace inspect-policy policy.onnx --format onnx --detailed
              marace replay adversarial.json --step --highlight-critical
              marace check-spec safety.yaml --env env.yaml
        """),
    )
    parser.add_argument(
        "--version", action="version", version=f"%(prog)s {_get_version()}"
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="enable debug logging"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="suppress informational output"
    )
    parser.add_argument(
        "--color", action="store_true", default=False,
        help="force colored output even when not a TTY",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # -- verify -------------------------------------------------------------
    p_verify = subparsers.add_parser(
        "verify", help="run the full verification pipeline",
    )
    p_verify.add_argument("--env", required=True, help="environment configuration file")
    p_verify.add_argument(
        "--policies", nargs="+", required=True, help="policy file path(s)"
    )
    p_verify.add_argument("--spec", default=None, help="safety specification file")
    p_verify.add_argument("-o", "--output", default=None, help="output directory")
    p_verify.add_argument(
        "--format", choices=_OUTPUT_FORMATS, default="text", help="output format"
    )
    p_verify.add_argument(
        "--timeout", type=float, default=300.0, help="timeout in seconds (default: 300)"
    )
    p_verify.add_argument(
        "--parallel", action=argparse.BooleanOptionalAction, default=True,
        help="enable/disable parallel execution (default: enabled)",
    )
    p_verify.add_argument(
        "--checkpoint-dir", default=None,
        help="directory for checkpoints (enables resume)",
    )
    p_verify.set_defaults(func=cmd_verify)

    # -- analyze-trace ------------------------------------------------------
    p_trace = subparsers.add_parser(
        "analyze-trace", help="analyze a recorded execution trace",
    )
    p_trace.add_argument("trace_file", help="path to the trace file")
    p_trace.add_argument("--spec", default=None, help="safety specification file")
    p_trace.add_argument("-o", "--output", default=None, help="output file path")
    p_trace.add_argument(
        "--format", choices=_OUTPUT_FORMATS, default="text", help="output format"
    )
    p_trace.set_defaults(func=cmd_analyze_trace)

    # -- benchmark ----------------------------------------------------------
    p_bench = subparsers.add_parser(
        "benchmark", help="run the benchmark suite",
    )
    p_bench.add_argument(
        "--suite",
        choices=("default", "scalability", "all"),
        default="default",
        help="benchmark suite to run",
    )
    p_bench.add_argument(
        "--num-agents", type=int, default=None, help="override number of agents"
    )
    p_bench.add_argument("-o", "--output", default=None, help="output directory")
    p_bench.add_argument(
        "--compare-baselines", action=argparse.BooleanOptionalAction, default=False,
        help="compare results against baseline methods",
    )
    p_bench.add_argument(
        "--timeout", type=float, default=600.0, help="timeout in seconds (default: 600)"
    )
    p_bench.set_defaults(func=cmd_benchmark)

    # -- report -------------------------------------------------------------
    p_report = subparsers.add_parser(
        "report", help="generate a report from saved results",
    )
    p_report.add_argument("results_file", help="path to the results JSON file")
    p_report.add_argument(
        "--format", choices=_OUTPUT_FORMATS, default="text", help="output format"
    )
    p_report.add_argument("-o", "--output", default=None, help="output file path")
    p_report.add_argument(
        "--include-certificates", action=argparse.BooleanOptionalAction, default=True,
        help="include verification certificates in the report",
    )
    p_report.set_defaults(func=cmd_report)

    # -- inspect-policy -----------------------------------------------------
    p_inspect = subparsers.add_parser(
        "inspect-policy", help="inspect and analyze a policy",
    )
    p_inspect.add_argument("policy_file", help="path to the policy file")
    p_inspect.add_argument(
        "--format",
        choices=("onnx", "pytorch", "custom"),
        default="onnx",
        help="policy file format",
    )
    detail_group = p_inspect.add_mutually_exclusive_group()
    detail_group.add_argument(
        "--summary", dest="detailed", action="store_false", default=False,
        help="show summary only (default)",
    )
    detail_group.add_argument(
        "--detailed", dest="detailed", action="store_true",
        help="show detailed analysis",
    )
    p_inspect.set_defaults(func=cmd_inspect_policy)

    # -- replay -------------------------------------------------------------
    p_replay = subparsers.add_parser(
        "replay", help="replay an adversarial trace interactively",
    )
    p_replay.add_argument("replay_file", help="path to the replay file")
    mode_group = p_replay.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--step", action="store_true", default=False,
        help="step through the trace one action at a time",
    )
    mode_group.add_argument(
        "--continuous", action="store_true", default=True,
        help="play the trace continuously (default)",
    )
    p_replay.add_argument(
        "--speed", type=float, default=1.0, help="playback speed multiplier (default: 1.0)"
    )
    p_replay.add_argument(
        "--highlight-critical", action="store_true", default=False,
        help="highlight critical race-condition steps",
    )
    p_replay.set_defaults(func=cmd_replay)

    # -- check-spec ---------------------------------------------------------
    p_check = subparsers.add_parser(
        "check-spec", help="validate a specification file",
    )
    p_check.add_argument("spec_file", help="path to the specification file")
    p_check.add_argument(
        "--env", default=None, help="environment config for compatibility check"
    )
    p_check.set_defaults(func=cmd_check_spec)

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for MARACE.

    Parameters
    ----------
    argv:
        Command-line arguments.  Defaults to ``sys.argv[1:]``.

    Returns
    -------
    int
        Exit code (0 for success, non-zero for failure).
    """
    global _color  # noqa: PLW0603

    parser = _build_parser()
    args = parser.parse_args(argv)

    # Reinitialise color output if --color was passed
    if args.color:
        _color = ColorOutput(force_color=True)

    _setup_logging(verbose=args.verbose, quiet=args.quiet)

    try:
        return args.func(args)
    except KeyboardInterrupt:
        print(
            f"\n{_color.warning('Interrupted by user')}", file=sys.stderr
        )
        return 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("marace.cli").exception("Unhandled error")
        print(_color.error(f"Fatal: {exc}"), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
