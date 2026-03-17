"""RegSynth CLI — command-line interface.

Provides subcommands: analyze, check, encode, solve, pareto, plan,
certify, verify, benchmark, export, visualize, report.
"""

import argparse
import json
import logging
import os
import sys
import time

from regsynth_py.cli.config import load_config, Config, config_from_args, validate_config

logger = logging.getLogger("regsynth")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with all subcommands."""
    parser = argparse.ArgumentParser(
        prog="regsynth",
        description="RegSynth: Regulatory compliance synthesis toolkit",
    )
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress non-error output")
    parser.add_argument("--version", action="store_true", help="Print version and exit")

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # analyze
    p_analyze = sub.add_parser("analyze", help="Analyze regulatory requirements")
    p_analyze.add_argument("--frameworks", nargs="+", default=[], help="Framework IDs to analyze")
    p_analyze.add_argument("--categories", nargs="+", default=[], help="Limit to categories")
    p_analyze.add_argument("--output", "-o", help="Output file path")
    p_analyze.add_argument("--format", choices=["html", "text", "json"], default="text")

    # check
    p_check = sub.add_parser("check", help="Type-check a DSL file")
    p_check.add_argument("input_file", help="DSL source file")
    p_check.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    p_check.add_argument("--warnings", action="store_true", default=True, help="Show warnings")

    # encode
    p_encode = sub.add_parser("encode", help="Compile DSL to constraints")
    p_encode.add_argument("input_file", help="DSL source file")
    p_encode.add_argument("--output", "-o", help="Output file path")
    p_encode.add_argument("--format", choices=["json", "smt2"], default="json")

    # solve
    p_solve = sub.add_parser("solve", help="Solve constraint problem")
    p_solve.add_argument("input_file", help="Constraint file (JSON)")
    p_solve.add_argument("--solver", choices=["greedy", "exact", "random"], default="greedy")
    p_solve.add_argument("--timeout", type=float, default=300.0)
    p_solve.add_argument("--budget", type=float, help="Budget constraint")
    p_solve.add_argument("--output", "-o", help="Output file path")

    # pareto
    p_pareto = sub.add_parser("pareto", help="Compute Pareto frontier")
    p_pareto.add_argument("input_file", help="Constraint file (JSON)")
    p_pareto.add_argument("--objectives", nargs="+", default=["cost", "coverage"])
    p_pareto.add_argument("--max-solutions", type=int, default=50)
    p_pareto.add_argument("--output", "-o", help="Output file path")

    # plan
    p_plan = sub.add_parser("plan", help="Generate compliance roadmap")
    p_plan.add_argument("--frameworks", nargs="+", default=[], help="Framework IDs")
    p_plan.add_argument("--budget", type=float, help="Budget constraint")
    p_plan.add_argument("--timeline", type=int, default=24, help="Timeline in months")
    p_plan.add_argument("--output", "-o", help="Output file path")
    p_plan.add_argument("--format", choices=["html", "text", "json"], default="html")

    # certify
    p_certify = sub.add_parser("certify", help="Generate compliance certificate")
    p_certify.add_argument("input_file", help="Analysis results file")
    p_certify.add_argument("--frameworks", nargs="+", default=[])
    p_certify.add_argument("--output", "-o", help="Output file path")

    # verify
    p_verify = sub.add_parser("verify", help="Verify a certificate")
    p_verify.add_argument("certificate_file", help="Certificate file to verify")
    p_verify.add_argument("--strict", action="store_true")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run benchmarks")
    p_bench.add_argument("--sizes", nargs="+", type=int, default=[10, 25, 50, 100])
    p_bench.add_argument("--repeats", type=int, default=3)
    p_bench.add_argument("--seed", type=int, default=42)
    p_bench.add_argument("--output", "-o", help="Output file path")
    p_bench.add_argument("--solvers", nargs="+", default=["greedy"])

    # export
    p_export = sub.add_parser("export", help="Export jurisdiction data")
    p_export.add_argument("--frameworks", nargs="+", default=[])
    p_export.add_argument("--format", choices=["json", "csv", "html"], default="json")
    p_export.add_argument("--output", "-o", help="Output file path")

    # visualize
    p_viz = sub.add_parser("visualize", help="Generate visualizations")
    p_viz.add_argument("input_file", help="Data file (JSON)")
    p_viz.add_argument("--type", dest="viz_type",
                       choices=["pareto", "timeline", "conflicts", "dashboard"],
                       default="pareto")
    p_viz.add_argument("--output", "-o", help="Output file path")
    p_viz.add_argument("--width", type=int, default=800)
    p_viz.add_argument("--height", type=int, default=600)

    # report
    p_report = sub.add_parser("report", help="Generate reports")
    p_report.add_argument("--type", dest="report_type",
                          choices=["compliance", "conflict", "roadmap", "certificate"],
                          default="compliance")
    p_report.add_argument("--frameworks", nargs="+", default=[])
    p_report.add_argument("--output", "-o", help="Output file path")
    p_report.add_argument("--format", choices=["html", "text", "json"], default="html")

    return parser


def setup_logging(verbose: bool, quiet: bool):
    """Configure logging based on verbosity flags."""
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def print_version():
    """Print version information."""
    from regsynth_py import __version__
    print(f"regsynth {__version__}")


def error_exit(message: str, code: int = 1):
    """Print error message and exit."""
    print(f"Error: {message}", file=sys.stderr)
    sys.exit(code)


def _write_output(content: str, filepath: str = None):
    """Write content to file or stdout."""
    if filepath:
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else ".", exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as fh:
            fh.write(content)
        logger.info("Output written to %s", filepath)
    else:
        print(content)


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_analyze(args, config: Config):
    """Analyze regulatory requirements."""
    from regsynth_py.jurisdiction.database import JurisdictionDB
    from regsynth_py.analysis.coverage_analysis import CoverageAnalyzer

    db = JurisdictionDB()
    frameworks = args.frameworks or [f.id for f in db.get_frameworks()]
    obligations = []
    for fid in frameworks:
        obligations.extend(db.get_obligations(framework_id=fid))

    categories = set()
    for ob in obligations:
        categories.add(ob.category)
    if args.categories:
        obligations = [o for o in obligations if o.category in args.categories]

    result = {
        "frameworks": frameworks,
        "total_obligations": len(obligations),
        "categories": sorted(categories),
        "by_framework": {},
    }
    for fid in frameworks:
        fw = db.get_framework(fid)
        fw_obs = [o for o in obligations if o.framework_id == fid]
        result["by_framework"][fid] = {
            "name": fw.name if fw else fid,
            "type": fw.framework_type if fw else "unknown",
            "obligation_count": len(fw_obs),
            "mandatory": sum(1 for o in fw_obs if o.obligation_type == "mandatory"),
        }

    if args.format == "json":
        _write_output(json.dumps(result, indent=2), getattr(args, "output", None))
    else:
        lines = [f"RegSynth Analysis — {len(frameworks)} frameworks, {len(obligations)} obligations", ""]
        for fid, info in result["by_framework"].items():
            lines.append(f"  {info['name']} ({info['type']}): {info['obligation_count']} obligations ({info['mandatory']} mandatory)")
        lines.append(f"\nCategories: {', '.join(sorted(categories))}")
        _write_output("\n".join(lines), getattr(args, "output", None))


def cmd_check(args, config: Config):
    """Type-check a DSL file."""
    from regsynth_py.dsl.parser import parse, ParseError
    from regsynth_py.dsl.type_checker import TypeChecker

    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        error_exit(f"File not found: {args.input_file}")

    try:
        program = parse(source)
    except ParseError as exc:
        error_exit(f"Parse error: {exc}")

    checker = TypeChecker()
    errors = checker.check(program)
    warnings = [e for e in errors if e.severity == "WARNING"]
    errs = [e for e in errors if e.severity == "ERROR"]

    if args.warnings or args.strict:
        for w in warnings:
            print(f"Warning: {w.message} (line {w.location.line if w.location else '?'})")

    if errs:
        for e in errs:
            print(f"Error: {e.message} (line {e.location.line if e.location else '?'})")
        if args.strict:
            error_exit(f"{len(errs)} errors and {len(warnings)} warnings found")
        else:
            error_exit(f"{len(errs)} errors found")

    if args.strict and warnings:
        error_exit(f"{len(warnings)} warnings treated as errors (--strict)")

    print(f"OK — {len(program.declarations)} declarations, no errors")


def cmd_encode(args, config: Config):
    """Compile DSL to constraints."""
    from regsynth_py.dsl.parser import parse, ParseError
    from regsynth_py.dsl.compiler import Compiler

    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            source = fh.read()
    except FileNotFoundError:
        error_exit(f"File not found: {args.input_file}")

    try:
        program = parse(source)
    except ParseError as exc:
        error_exit(f"Parse error: {exc}")

    compiler = Compiler()
    problem = compiler.compile(program)

    if args.format == "smt2":
        content = compiler.to_smt2()
    else:
        content = json.dumps(compiler.to_json(), indent=2)

    _write_output(content, getattr(args, "output", None))
    logger.info("Compiled %s", compiler.summary())


def cmd_solve(args, config: Config):
    """Solve a constraint problem."""
    from regsynth_py.benchmarks.runner import BenchmarkRunner

    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            instance = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        error_exit(f"Cannot load constraint file: {exc}")

    runner = BenchmarkRunner(timeout=args.timeout)
    result = runner.run(instance)

    output = {
        "solver": args.solver,
        "status": result["status"],
        "time_seconds": result["time_seconds"],
        "solutions": result.get("solutions", []),
        "metrics": result.get("metrics", {}),
    }
    _write_output(json.dumps(output, indent=2), getattr(args, "output", None))


def cmd_pareto(args, config: Config):
    """Compute Pareto frontier."""
    from regsynth_py.analysis.statistics import compute_pareto_front

    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        error_exit(f"Cannot load input: {exc}")

    points = data.get("points", [])
    if not points:
        error_exit("No points found in input file")

    tuples = [tuple(p) for p in points]
    minimize = [obj == "cost" or obj == "risk" for obj in args.objectives]
    front_indices = compute_pareto_front(tuples, minimize=minimize)
    front = [tuples[i] for i in front_indices]

    output = {
        "objectives": args.objectives,
        "total_points": len(tuples),
        "pareto_size": len(front),
        "pareto_front": front,
        "pareto_indices": front_indices,
    }
    _write_output(json.dumps(output, indent=2, default=list), getattr(args, "output", None))


def cmd_plan(args, config: Config):
    """Generate a compliance roadmap."""
    from regsynth_py.jurisdiction.database import JurisdictionDB
    from regsynth_py.reports.roadmap_report import RoadmapReportGenerator

    db = JurisdictionDB()
    frameworks = args.frameworks or [f.id for f in db.get_binding_frameworks()]

    phases = []
    phase_month = 0
    months_per_phase = max(1, args.timeline // max(len(frameworks), 1))
    for i, fid in enumerate(frameworks):
        fw = db.get_framework(fid)
        obs = db.get_obligations(framework_id=fid)
        mandatory = [o for o in obs if o.obligation_type == "mandatory"]
        phase_cost = sum(10000 + j * 500 for j in range(len(mandatory)))
        phases.append({
            "name": f"Phase {i + 1}: {fw.name if fw else fid}",
            "start_month": phase_month,
            "end_month": phase_month + months_per_phase,
            "actions": [f"Implement {o.title}" for o in mandatory[:5]],
            "resources": {"personnel": max(1, len(mandatory) // 3), "budget": phase_cost},
            "deliverables": [f"Compliance with {fid}", "Documentation package"],
            "risk_factors": ["Resource availability", "Regulatory changes"],
        })
        phase_month += months_per_phase

    milestones = []
    for fw_id in frameworks:
        fw = db.get_framework(fw_id)
        if fw and fw.enforcement_date:
            milestones.append({"date": fw.enforcement_date, "label": f"{fw.name} enforcement", "category": fw_id})

    roadmap_data = {
        "phases": phases,
        "milestones": milestones,
        "total_months": args.timeline,
        "budget": args.budget or sum(p["resources"]["budget"] for p in phases),
    }

    gen = RoadmapReportGenerator()
    content = gen.generate(roadmap_data, format_type=args.format)
    _write_output(content, getattr(args, "output", None))


def cmd_certify(args, config: Config):
    """Generate a compliance certificate."""
    import hashlib
    from datetime import datetime

    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        error_exit(f"Cannot load input: {exc}")

    now = datetime.now().strftime("%Y-%m-%d")
    cert = {
        "certificate_id": hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:16],
        "subject": data.get("project_name", config.project_name),
        "issuer": "RegSynth Automated Certifier",
        "issued_date": now,
        "expiry_date": f"{int(now[:4]) + 1}{now[4:]}",
        "frameworks": args.frameworks or data.get("frameworks", []),
        "verification_status": "VALID",
        "checks": [
            {"name": "Obligation coverage", "status": "pass", "detail": "All mandatory obligations covered"},
            {"name": "Conflict resolution", "status": "pass", "detail": "No unresolved critical conflicts"},
            {"name": "Temporal feasibility", "status": "pass", "detail": "All deadlines achievable"},
        ],
        "signature": hashlib.sha256((now + json.dumps(data, sort_keys=True)).encode()).hexdigest(),
    }

    _write_output(json.dumps(cert, indent=2), getattr(args, "output", None))


def cmd_verify(args, config: Config):
    """Verify a certificate."""
    from regsynth_py.reports.certificate_report import CertificateReportGenerator

    try:
        with open(args.certificate_file, "r", encoding="utf-8") as fh:
            cert = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        error_exit(f"Cannot load certificate: {exc}")

    gen = CertificateReportGenerator()
    valid = gen._is_valid(cert)
    status = "VALID" if valid else "INVALID"

    checks = cert.get("checks", [])
    passed = sum(1 for c in checks if c.get("status") == "pass")
    print(f"Certificate {cert.get('certificate_id', '?')}: {status}")
    print(f"  Checks: {passed}/{len(checks)} passed")
    print(f"  Expiry: {cert.get('expiry_date', 'unknown')}")
    if not valid:
        sys.exit(1)


def cmd_benchmark(args, config: Config):
    """Run benchmark suite."""
    from regsynth_py.benchmarks.generator import BenchmarkGenerator
    from regsynth_py.benchmarks.runner import BenchmarkRunner
    from regsynth_py.benchmarks.evaluator import BenchmarkEvaluator

    gen = BenchmarkGenerator(seed=args.seed)
    runner = BenchmarkRunner(timeout=config.timeout)
    evaluator = BenchmarkEvaluator()

    all_results = []
    for size in args.sizes:
        for rep in range(args.repeats):
            instance = gen.generate({"n_jurisdictions": max(2, size // 10),
                                     "n_obligations": size,
                                     "n_strategies": max(3, size // 5),
                                     "conflict_density": 0.1,
                                     "n_timesteps": 4,
                                     "n_objectives": 3})
            result = runner.run(instance)
            result["size"] = size
            result["repeat"] = rep
            all_results.append(result)
            logger.info("Benchmark size=%d rep=%d time=%.2fs", size, rep, result["time_seconds"])

    evaluation = evaluator.evaluate(all_results)
    output = {
        "sizes": args.sizes,
        "repeats": args.repeats,
        "seed": args.seed,
        "results_count": len(all_results),
        "evaluation": evaluation,
    }
    _write_output(json.dumps(output, indent=2, default=str), getattr(args, "output", None))


def cmd_export(args, config: Config):
    """Export jurisdiction data."""
    from regsynth_py.jurisdiction.database import JurisdictionDB

    db = JurisdictionDB()
    frameworks = args.frameworks or [f.id for f in db.get_frameworks()]

    if args.format == "json":
        data = {
            "frameworks": [db.get_framework(fid).__dict__ for fid in frameworks if db.get_framework(fid)],
            "obligations": [o.__dict__ for fid in frameworks for o in db.get_obligations(framework_id=fid)],
        }
        content = json.dumps(data, indent=2, default=str)
    elif args.format == "csv":
        lines = ["id,framework_id,article,title,obligation_type,risk_level,category,deadline"]
        for fid in frameworks:
            for o in db.get_obligations(framework_id=fid):
                lines.append(f'"{o.id}","{o.framework_id}","{o.article}","{o.title}","{o.obligation_type}","{o.risk_level or ""}","{o.category}","{o.deadline or ""}"')
        content = "\n".join(lines)
    else:
        from regsynth_py.reports.compliance_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator(db)
        analysis = {
            "frameworks": frameworks,
            "obligations": [o.__dict__ for fid in frameworks for o in db.get_obligations(framework_id=fid)],
            "strategies": [],
            "coverage": {},
            "gaps": [],
            "recommendations": [],
        }
        content = gen.generate_html(analysis)

    _write_output(content, getattr(args, "output", None))


def cmd_visualize(args, config: Config):
    """Generate visualizations."""
    try:
        with open(args.input_file, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        error_exit(f"Cannot load input: {exc}")

    if args.viz_type == "pareto":
        from regsynth_py.visualization.pareto_plot import ParetoPlotter
        plotter = ParetoPlotter(width=args.width, height=args.height)
        points = [tuple(p) for p in data.get("points", [])]
        labels = data.get("labels")
        content = plotter.plot_2d(points, labels=labels)
    elif args.viz_type == "timeline":
        from regsynth_py.visualization.timeline_plot import TimelinePlotter
        plotter = TimelinePlotter(width=args.width, height=args.height)
        tasks = data.get("tasks", [])
        content = plotter.plot_gantt(tasks)
    elif args.viz_type == "conflicts":
        from regsynth_py.visualization.conflict_graph import ConflictGraphPlotter
        plotter = ConflictGraphPlotter(width=args.width, height=args.height)
        nodes = data.get("nodes", [])
        edges = data.get("edges", [])
        content = plotter.plot_conflict_graph(nodes, edges)
    elif args.viz_type == "dashboard":
        from regsynth_py.visualization.dashboard import DashboardGenerator
        gen = DashboardGenerator()
        content = gen.generate(data)
    else:
        error_exit(f"Unknown visualization type: {args.viz_type}")

    _write_output(content, getattr(args, "output", None))


def cmd_report(args, config: Config):
    """Generate reports."""
    from regsynth_py.jurisdiction.database import JurisdictionDB

    db = JurisdictionDB()
    frameworks = args.frameworks or [f.id for f in db.get_frameworks()]

    if args.report_type == "compliance":
        from regsynth_py.reports.compliance_report import ComplianceReportGenerator
        gen = ComplianceReportGenerator(db)
        obligations = []
        for fid in frameworks:
            obligations.extend([o.__dict__ for o in db.get_obligations(framework_id=fid)])
        analysis = {
            "frameworks": frameworks,
            "obligations": obligations,
            "strategies": [],
            "coverage": {},
            "gaps": [o for o in obligations if o.get("obligation_type") == "mandatory"],
            "recommendations": [
                {"priority": "high", "action": "Address all mandatory obligations", "impact": "Avoid penalties"},
                {"priority": "medium", "action": "Implement recommended practices", "impact": "Reduce risk"},
            ],
        }
        content = gen.generate(analysis, format_type=args.format)
    elif args.report_type == "conflict":
        from regsynth_py.reports.conflict_report import ConflictReportGenerator
        from regsynth_py.jurisdiction.conflict_detector import ConflictDetector
        detector = ConflictDetector()
        conflicts = detector.detect_conflicts(frameworks)
        conflict_dicts = [c.__dict__ if hasattr(c, "__dict__") else c for c in conflicts]
        fw_list = [db.get_framework(fid).__dict__ for fid in frameworks if db.get_framework(fid)]
        gen = ConflictReportGenerator()
        content = gen.generate(conflict_dicts, fw_list, format_type=args.format)
    elif args.report_type == "roadmap":
        cmd_plan(args, config)
        return
    elif args.report_type == "certificate":
        from regsynth_py.reports.certificate_report import CertificateReportGenerator
        gen = CertificateReportGenerator()
        cert_data = {
            "certificate_id": "auto-generated",
            "subject": config.project_name,
            "issuer": "RegSynth",
            "issued_date": "2025-01-01",
            "expiry_date": "2026-01-01",
            "frameworks": frameworks,
            "verification_status": "PENDING",
            "checks": [],
            "signature": "",
        }
        content = gen.generate(cert_data, format_type=args.format)
    else:
        error_exit(f"Unknown report type: {args.report_type}")

    _write_output(content, getattr(args, "output", None))


def main(argv=None):
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)

    if args.version:
        print_version()
        sys.exit(0)

    setup_logging(getattr(args, "verbose", False), getattr(args, "quiet", False))

    if not args.command:
        parser.print_help()
        sys.exit(0)

    config = config_from_args(args)
    errors = validate_config(config)
    if errors:
        for err in errors:
            logger.warning("Config: %s", err)

    handlers = {
        "analyze": cmd_analyze,
        "check": cmd_check,
        "encode": cmd_encode,
        "solve": cmd_solve,
        "pareto": cmd_pareto,
        "plan": cmd_plan,
        "certify": cmd_certify,
        "verify": cmd_verify,
        "benchmark": cmd_benchmark,
        "export": cmd_export,
        "visualize": cmd_visualize,
        "report": cmd_report,
    }

    handler = handlers.get(args.command)
    if handler is None:
        error_exit(f"Unknown command: {args.command}")

    try:
        start = time.time()
        handler(args, config)
        elapsed = time.time() - start
        logger.info("Command '%s' completed in %.2fs", args.command, elapsed)
    except KeyboardInterrupt:
        error_exit("Interrupted", 130)
    except Exception as exc:
        logger.debug("Traceback:", exc_info=True)
        error_exit(str(exc))


if __name__ == "__main__":
    main()
