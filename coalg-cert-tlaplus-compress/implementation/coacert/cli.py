"""
CLI interface for CoaCert-TLA.

Provides subcommands for parsing, exploring, compressing, verifying,
and benchmarking TLA-lite specifications through coalgebraic bisimulation
quotient compression with Merkle-hashed witnesses.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger("coacert")

# ---------------------------------------------------------------------------
# ANSI colour helpers
# ---------------------------------------------------------------------------

_NO_COLOR = os.environ.get("NO_COLOR") is not None or not sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    if _NO_COLOR:
        return text
    return f"\033[{code}m{text}\033[0m"


def green(t: str) -> str:
    return _c("32", t)


def red(t: str) -> str:
    return _c("31", t)


def yellow(t: str) -> str:
    return _c("33", t)


def cyan(t: str) -> str:
    return _c("36", t)


def bold(t: str) -> str:
    return _c("1", t)


def dim(t: str) -> str:
    return _c("2", t)


# ---------------------------------------------------------------------------
# Simple progress indicator
# ---------------------------------------------------------------------------

class ProgressBar:
    """Minimal progress bar for terminal output."""

    def __init__(self, total: int, desc: str = "", width: int = 40, enabled: bool = True):
        self.total = max(total, 1)
        self.current = 0
        self.desc = desc
        self.width = width
        self.enabled = enabled and sys.stderr.isatty() and not _NO_COLOR
        self._start = time.monotonic()

    def update(self, n: int = 1) -> None:
        self.current = min(self.current + n, self.total)
        if not self.enabled:
            return
        frac = self.current / self.total
        filled = int(self.width * frac)
        bar = "█" * filled + "░" * (self.width - filled)
        elapsed = time.monotonic() - self._start
        rate = self.current / elapsed if elapsed > 0 else 0
        line = f"\r  {self.desc} [{bar}] {self.current}/{self.total} ({rate:.1f} it/s)"
        sys.stderr.write(line)
        sys.stderr.flush()

    def finish(self) -> None:
        if self.enabled:
            self.current = self.total
            self.update(0)
            sys.stderr.write("\n")
            sys.stderr.flush()


# ---------------------------------------------------------------------------
# Output formatting helpers
# ---------------------------------------------------------------------------

def _write_output(data: Any, args: argparse.Namespace) -> None:
    """Write structured data to --output or stdout in the chosen --format."""
    fmt = getattr(args, "format", "text")
    dest = getattr(args, "output", None)

    if fmt == "json":
        text = json.dumps(data, indent=2, default=str)
    elif fmt == "latex":
        text = _to_latex_table(data) if isinstance(data, (list, dict)) else str(data)
    else:
        text = _to_text(data)

    if dest:
        Path(dest).write_text(text + "\n")
        print(dim(f"Output written to {dest}"))
    else:
        print(text)


def _to_text(data: Any, indent: int = 0) -> str:
    prefix = "  " * indent
    if isinstance(data, dict):
        lines = []
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{bold(str(k))}:")
                lines.append(_to_text(v, indent + 1))
            else:
                lines.append(f"{prefix}{bold(str(k))}: {v}")
        return "\n".join(lines)
    if isinstance(data, list):
        return "\n".join(f"{prefix}- {_to_text(item, indent + 1).strip()}" for item in data)
    return f"{prefix}{data}"


def _to_latex_table(data: Any) -> str:
    if isinstance(data, dict):
        rows = list(data.items())
    elif isinstance(data, list) and data and isinstance(data[0], dict):
        headers = list(data[0].keys())
        header_line = " & ".join(f"\\textbf{{{h}}}" for h in headers) + " \\\\"
        body = "\n".join(
            " & ".join(str(row.get(h, "")) for h in headers) + " \\\\"
            for row in data
        )
        return (
            "\\begin{tabular}{" + "l" * len(headers) + "}\n\\hline\n"
            + header_line + "\n\\hline\n" + body + "\n\\hline\n\\end{tabular}"
        )
    else:
        rows = [(i, v) for i, v in enumerate(data)] if isinstance(data, list) else [("value", data)]
    header_line = "\\textbf{Key} & \\textbf{Value} \\\\"
    body = "\n".join(f"{k} & {v} \\\\" for k, v in rows)
    return (
        "\\begin{tabular}{ll}\n\\hline\n"
        + header_line + "\n\\hline\n" + body + "\n\\hline\n\\end{tabular}"
    )


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------

def cmd_parse(args: argparse.Namespace) -> int:
    """Parse a TLA-lite spec and optionally type-check / pretty-print."""
    from coacert.parser import parse, TypeChecker, PrettyPrinter

    path = Path(args.file)
    if not path.exists():
        print(red(f"Error: file not found: {path}"))
        return 1

    source = path.read_text()
    print(dim(f"Parsing {path.name} ({len(source)} chars)…"))

    try:
        module = parse(source)
    except Exception as exc:
        print(red(f"Parse error: {exc}"))
        if args.verbose:
            traceback.print_exc()
        return 1

    print(green("✓ Parsed successfully"))

    if args.check_types:
        try:
            tc = TypeChecker()
            tc.check(module)
            print(green("✓ Type check passed"))
        except Exception as exc:
            print(red(f"Type error: {exc}"))
            if args.verbose:
                traceback.print_exc()
            return 1

    if args.pretty_print:
        pp = PrettyPrinter()
        output = pp.format(module)
        print()
        print(bold("--- Pretty-printed spec ---"))
        print(output)
    else:
        summary = {
            "file": str(path),
            "module_name": getattr(module, "name", str(path.stem)),
            "definitions": len(getattr(module, "definitions", [])),
            "properties": len(getattr(module, "properties", [])),
            "variables": len(getattr(module, "variables", [])),
        }
        _write_output(summary, args)

    return 0


def cmd_explore(args: argparse.Namespace) -> int:
    """Explore the state space of a TLA-lite spec."""
    from coacert.parser import parse
    from coacert.semantics import compute_initial_states, compute_successors
    from coacert.explorer import ExplicitStateExplorer, TransitionGraph

    # Resolve spec source
    source = _resolve_spec_source(args)
    if source is None:
        return 1

    module = parse(source)
    print(dim("Building state space…"))

    explorer = ExplicitStateExplorer(
        module,
        max_states=args.max_states,
        max_depth=args.max_depth,
        strategy=args.strategy,
    )

    progress = ProgressBar(args.max_states, desc="Exploring", enabled=args.verbose)

    def on_state(count: int) -> None:
        progress.update(1)

    try:
        graph = explorer.explore(callback=on_state)
    except Exception as exc:
        print(red(f"Exploration error: {exc}"))
        if args.verbose:
            traceback.print_exc()
        return 1
    finally:
        progress.finish()

    stats = graph.stats() if hasattr(graph, "stats") else {}
    num_states = stats.get("states", len(graph.nodes) if hasattr(graph, "nodes") else 0)
    num_trans = stats.get("transitions", len(graph.edges) if hasattr(graph, "edges") else 0)

    result = {
        "states": num_states,
        "transitions": num_trans,
        "max_depth": stats.get("max_depth", args.max_depth),
        "strategy": args.strategy,
        "deadlocks": stats.get("deadlocks", 0),
    }

    print(green(f"✓ Explored {num_states} states, {num_trans} transitions"))

    if args.output_graph:
        _export_graph(graph, args.output_graph)
        print(dim(f"Graph written to {args.output_graph}"))

    _write_output(result, args)
    return 0


def cmd_compress(args: argparse.Namespace) -> int:
    """Run the full compression pipeline: explore → learn → quotient → witness."""
    from coacert.pipeline import Pipeline, PipelineConfig

    source = _resolve_spec_source(args)
    if source is None:
        return 1

    config = PipelineConfig(
        conformance_depth=args.conformance_depth,
        max_learning_rounds=args.max_rounds,
        max_states=getattr(args, "max_states", 100_000),
        verbose=args.verbose,
    )

    pipeline = Pipeline(config)

    def on_stage(name: str, pct: float) -> None:
        if args.verbose:
            sys.stderr.write(f"\r  {cyan(name)}: {pct:.0%}")
            sys.stderr.flush()

    print(bold("CoaCert compression pipeline"))
    print(dim("─" * 40))

    try:
        result = pipeline.run(source, stage_callback=on_stage)
    except Exception as exc:
        print(red(f"\nPipeline error: {exc}"))
        if args.verbose:
            traceback.print_exc()
        return 1

    if args.verbose:
        sys.stderr.write("\n")

    # Summary
    orig = result.original_states
    quot = result.quotient_states
    ratio = quot / orig if orig else 0
    print(green(f"✓ Compression complete"))
    print(f"  Original states:  {orig}")
    print(f"  Quotient states:  {quot}")
    print(f"  Compression ratio: {ratio:.2%}")
    print(f"  Learning rounds:  {result.learning_rounds}")
    print(f"  Witness size:     {result.witness_size_bytes} bytes")
    print(f"  Time:             {result.elapsed_seconds:.2f}s")

    if args.output_witness:
        result.write_witness(args.output_witness)
        print(dim(f"  Witness → {args.output_witness}"))

    if args.output_quotient:
        result.write_quotient(args.output_quotient)
        print(dim(f"  Quotient → {args.output_quotient}"))

    summary = {
        "original_states": orig,
        "quotient_states": quot,
        "compression_ratio": round(ratio, 4),
        "learning_rounds": result.learning_rounds,
        "witness_size_bytes": result.witness_size_bytes,
        "elapsed_seconds": round(result.elapsed_seconds, 3),
        "witness_verified": result.witness_verified,
        "properties_preserved": result.properties_preserved,
    }
    _write_output(summary, args)
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Verify a bisimulation witness file."""
    from coacert.verifier import verify_witness, VerificationReport

    path = Path(args.witness_file)
    if not path.exists():
        print(red(f"Error: witness file not found: {path}"))
        return 1

    print(dim(f"Verifying witness: {path.name}…"))

    try:
        report: VerificationReport = verify_witness(str(path))
    except Exception as exc:
        print(red(f"Verification error: {exc}"))
        if args.verbose:
            traceback.print_exc()
        return 1

    verdict_ok = getattr(report, "verdict", None)
    # VerificationReport may use .passed, .verdict, or .ok
    passed = False
    if isinstance(verdict_ok, bool):
        passed = verdict_ok
    elif hasattr(report, "passed"):
        passed = report.passed
    elif hasattr(report, "ok"):
        passed = report.ok
    else:
        passed = str(verdict_ok).lower() in ("pass", "passed", "true", "ok")

    if passed:
        print(green("✓ Witness verified successfully"))
    else:
        print(red("✗ Witness verification FAILED"))

    if args.verbose and hasattr(report, "details"):
        for key, val in report.details.items():
            status = green("✓") if val else red("✗")
            print(f"  {status} {key}")

    if args.spot_check:
        print(dim("Running spot-check sampling…"))
        try:
            from coacert.verifier import HashChainVerifier
            hv = HashChainVerifier()
            sample_ok = hv.spot_check(str(path), samples=50)
            if sample_ok:
                print(green("  ✓ Spot-check passed (50 samples)"))
            else:
                print(red("  ✗ Spot-check failed"))
        except Exception as exc:
            print(yellow(f"  ⚠ Spot-check unavailable: {exc}"))

    result = {
        "file": str(path),
        "verified": passed,
        "checks": getattr(report, "details", {}),
    }
    _write_output(result, args)
    return 0 if passed else 1


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run evaluation benchmarks over a set of specs."""
    from coacert.evaluation import BenchmarkRunner, MetricsCollector
    from coacert.specs import SpecRegistry
    from coacert.pipeline import Pipeline, PipelineConfig

    registry = SpecRegistry()
    available = registry.list_specs() if hasattr(registry, "list_specs") else []

    specs = args.specs if args.specs else available
    if not specs:
        print(yellow("No specs to benchmark. Use --specs or ensure built-in specs are available."))
        return 1

    runs = args.runs
    print(bold(f"Benchmarking {len(specs)} spec(s), {runs} run(s) each"))
    print(dim("─" * 50))

    config = PipelineConfig(verbose=False)
    runner = BenchmarkRunner()
    collector = MetricsCollector()

    all_results: List[Dict[str, Any]] = []

    for spec_name in specs:
        print(f"\n{cyan(spec_name)}")

        try:
            source = registry.get_source(spec_name) if hasattr(registry, "get_source") else None
            if source is None:
                spec_cls = registry.get(spec_name) if hasattr(registry, "get") else None
                if spec_cls is None:
                    print(yellow(f"  ⚠ Spec '{spec_name}' not found, skipping"))
                    continue
                builder = spec_cls()
                source = builder.to_source() if hasattr(builder, "to_source") else str(builder.build())
        except Exception as exc:
            print(yellow(f"  ⚠ Could not load '{spec_name}': {exc}"))
            continue

        timings = []
        last_result = None

        progress = ProgressBar(runs, desc=spec_name, enabled=True)
        for r in range(runs):
            pipeline = Pipeline(config)
            t0 = time.monotonic()
            try:
                result = pipeline.run(source)
                elapsed = time.monotonic() - t0
                timings.append(elapsed)
                last_result = result
                collector.record(spec_name, r, elapsed, result)
            except Exception as exc:
                elapsed = time.monotonic() - t0
                timings.append(elapsed)
                print(yellow(f"    Run {r + 1} failed: {exc}"))
            progress.update(1)
        progress.finish()

        if timings:
            avg = sum(timings) / len(timings)
            best = min(timings)
            row: Dict[str, Any] = {
                "spec": spec_name,
                "runs": runs,
                "avg_time_s": round(avg, 3),
                "best_time_s": round(best, 3),
            }
            if last_result:
                row["original_states"] = last_result.original_states
                row["quotient_states"] = last_result.quotient_states
                ratio = last_result.quotient_states / last_result.original_states if last_result.original_states else 0
                row["compression"] = f"{ratio:.2%}"
                row["witness_verified"] = last_result.witness_verified
            all_results.append(row)
            print(f"  avg={avg:.3f}s  best={best:.3f}s  "
                  f"states={row.get('original_states', '?')}→{row.get('quotient_states', '?')}")

    print(f"\n{bold('Results')}")
    print(dim("─" * 50))
    _write_output(all_results, args)

    if args.output_report:
        fmt = getattr(args, "format", "json")
        if fmt == "json":
            Path(args.output_report).write_text(json.dumps(all_results, indent=2, default=str))
        elif fmt == "latex":
            Path(args.output_report).write_text(_to_latex_table(all_results))
        else:
            Path(args.output_report).write_text(_to_text(all_results))
        print(dim(f"Report written to {args.output_report}"))

    return 0


def cmd_info(args: argparse.Namespace) -> int:
    """Show information about built-in specs."""
    from coacert.specs import SpecRegistry

    registry = SpecRegistry()
    available = registry.list_specs() if hasattr(registry, "list_specs") else []

    if args.spec:
        name = args.spec
        meta = registry.get_metadata(name) if hasattr(registry, "get_metadata") else None
        if meta is None:
            # Try to load directly
            spec_cls = registry.get(name) if hasattr(registry, "get") else None
            if spec_cls is None:
                print(red(f"Unknown spec: {name}"))
                print(f"Available: {', '.join(available) if available else '(none)'}")
                return 1
            meta = {
                "name": name,
                "class": spec_cls.__name__ if hasattr(spec_cls, "__name__") else str(spec_cls),
            }
        print(bold(f"Spec: {name}"))
        _write_output(meta, args)
    else:
        print(bold("Available built-in specs"))
        print(dim("─" * 30))
        if not available:
            print(yellow("No built-in specs found."))
            return 0
        for name in available:
            meta = registry.get_metadata(name) if hasattr(registry, "get_metadata") else {}
            desc = meta.get("description", "") if isinstance(meta, dict) else ""
            print(f"  {cyan(name)}: {desc}" if desc else f"  {cyan(name)}")

    return 0


# ---------------------------------------------------------------------------
# Spec resolution helper
# ---------------------------------------------------------------------------

def _resolve_spec_source(args: argparse.Namespace) -> Optional[str]:
    """Return TLA-lite source from --file or --spec."""
    if hasattr(args, "file") and args.file:
        path = Path(args.file)
        if not path.exists():
            print(red(f"Error: file not found: {path}"))
            return None
        return path.read_text()

    if hasattr(args, "spec") and args.spec:
        from coacert.specs import SpecRegistry
        registry = SpecRegistry()
        try:
            if hasattr(registry, "get_source"):
                source = registry.get_source(args.spec)
                if source:
                    return source
            spec_cls = registry.get(args.spec) if hasattr(registry, "get") else None
            if spec_cls is None:
                print(red(f"Unknown built-in spec: {args.spec}"))
                return None
            builder = spec_cls()
            return builder.to_source() if hasattr(builder, "to_source") else str(builder.build())
        except Exception as exc:
            print(red(f"Error loading spec '{args.spec}': {exc}"))
            return None

    print(red("Error: provide --file or --spec"))
    return None


def _export_graph(graph: Any, path: str) -> None:
    """Export a TransitionGraph to a JSON file."""
    import networkx as nx

    if isinstance(graph, nx.DiGraph):
        data = nx.node_link_data(graph)
    elif hasattr(graph, "to_networkx"):
        data = nx.node_link_data(graph.to_networkx())
    elif hasattr(graph, "to_dict"):
        data = graph.to_dict()
    else:
        data = {"nodes": list(getattr(graph, "nodes", [])),
                "edges": list(getattr(graph, "edges", []))}

    Path(path).write_text(json.dumps(data, indent=2, default=str))


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="coacert",
        description="CoaCert-TLA: Coalgebraic Certified Compression for TLA+ Specifications",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  coacert parse --file spec.tla --check-types
  coacert explore --spec TwoPhaseCommit --max-states 50000
  coacert compress --file spec.tla --output-witness wit.json
  coacert verify --witness-file wit.json --spot-check
  coacert benchmark --specs TwoPhaseCommit Paxos --runs 5
  coacert info --spec TwoPhaseCommit
""",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--output", "-o", type=str, default=None, help="Write output to file")
    parser.add_argument(
        "--format", "-f",
        choices=["text", "json", "latex"],
        default="text",
        help="Output format (default: text)",
    )

    subs = parser.add_subparsers(dest="command", help="Available subcommands")

    # -- parse ---------------------------------------------------------------
    p_parse = subs.add_parser("parse", help="Parse a TLA-lite specification file")
    p_parse.add_argument("--file", required=True, help="Path to TLA-lite spec file")
    p_parse.add_argument("--check-types", action="store_true", help="Run type checker on the AST")
    p_parse.add_argument("--pretty-print", action="store_true", help="Pretty-print the parsed module")

    # -- explore -------------------------------------------------------------
    p_explore = subs.add_parser("explore", help="Explore state space of a TLA-lite spec")
    p_explore.add_argument("--file", default=None, help="Path to TLA-lite spec file")
    p_explore.add_argument("--spec", default=None, help="Name of built-in spec (e.g. TwoPhaseCommit)")
    p_explore.add_argument("--max-states", type=int, default=100_000, help="Maximum states to explore")
    p_explore.add_argument("--max-depth", type=int, default=1000, help="Maximum exploration depth")
    p_explore.add_argument(
        "--strategy",
        choices=["bfs", "dfs"],
        default="bfs",
        help="Exploration strategy (default: bfs)",
    )
    p_explore.add_argument("--output-graph", default=None, help="Export transition graph to JSON file")

    # -- compress ------------------------------------------------------------
    p_compress = subs.add_parser("compress", help="Run full compression pipeline")
    p_compress.add_argument("--file", default=None, help="Path to TLA-lite spec file")
    p_compress.add_argument("--spec", default=None, help="Name of built-in spec")
    p_compress.add_argument(
        "--conformance-depth", type=int, default=10,
        help="Depth for conformance-testing equivalence oracle (default: 10)",
    )
    p_compress.add_argument(
        "--max-rounds", type=int, default=1000,
        help="Maximum L* learning rounds (default: 1000)",
    )
    p_compress.add_argument(
        "--max-states", type=int, default=100_000,
        help="Maximum states for initial exploration (default: 100000)",
    )
    p_compress.add_argument("--output-witness", default=None, help="Write witness certificate to file")
    p_compress.add_argument("--output-quotient", default=None, help="Write quotient system to file")

    # -- verify --------------------------------------------------------------
    p_verify = subs.add_parser("verify", help="Verify a bisimulation witness certificate")
    p_verify.add_argument("--witness-file", required=True, help="Path to witness JSON file")
    p_verify.add_argument("--spot-check", action="store_true", help="Run random spot-check sampling")

    # -- benchmark -----------------------------------------------------------
    p_bench = subs.add_parser("benchmark", help="Run evaluation benchmarks")
    p_bench.add_argument(
        "--specs", nargs="*", default=None,
        help="Spec names to benchmark (default: all built-in specs)",
    )
    p_bench.add_argument("--runs", type=int, default=3, help="Number of runs per spec (default: 3)")
    p_bench.add_argument("--output-report", default=None, help="Write benchmark report to file")

    # -- info ----------------------------------------------------------------
    p_info = subs.add_parser("info", help="Show information about built-in specs")
    p_info.add_argument("--spec", default=None, help="Name of spec to inspect (omit to list all)")

    return parser


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

_DISPATCH = {
    "parse": cmd_parse,
    "explore": cmd_explore,
    "compress": cmd_compress,
    "verify": cmd_verify,
    "benchmark": cmd_benchmark,
    "info": cmd_info,
}


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    # Logging
    level = logging.DEBUG if args.verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    if not args.command:
        parser.print_help()
        return 0

    handler = _DISPATCH.get(args.command)
    if handler is None:
        print(red(f"Unknown command: {args.command}"))
        parser.print_help()
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        print(yellow("\nInterrupted"))
        return 130
    except Exception as exc:
        print(red(f"Unexpected error: {exc}"))
        if args.verbose:
            traceback.print_exc()
        print(dim("Run with --verbose for full traceback"))
        return 1


if __name__ == "__main__":
    sys.exit(main())
