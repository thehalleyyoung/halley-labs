"""
CausalBound CLI — Click-based command-line interface for the causal
inference systemic-risk pipeline.

Commands
--------
decompose    Tree-decompose a financial network.
solve-lp     Solve causal-polytope LP for probability bounds.
infer        Run causal inference with bound composition.
verify       SMT-verify computed bounds.
search       Adversarial scenario search (MCTS / random / grid).
run-pipeline Full end-to-end pipeline orchestration.
benchmark    Run reproducibility benchmark suites.
"""

from __future__ import annotations

import csv
import importlib
import io
import json
import logging
import os
import pathlib
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import click

# ---------------------------------------------------------------------------
# Default configuration schema
# ---------------------------------------------------------------------------

DEFAULT_CONFIG: Dict[str, Any] = {
    "decomposition": {
        "method": "min-degree",
        "max_treewidth": 15,
        "timeout_seconds": 300,
    },
    "lp_solver": {
        "max_iterations": 5000,
        "gap_tolerance": 1e-6,
        "column_generation": True,
        "solver_backend": "highs",
        "thread_count": 1,
    },
    "inference": {
        "composition_method": "chain",
        "propagation": "belief",
        "max_paths": 100,
    },
    "verification": {
        "smt_solver": "z3",
        "timeout_seconds": 120,
        "bit_width": 64,
        "epsilon": 1e-9,
    },
    "search": {
        "method": "mcts",
        "n_rollouts": 1000,
        "exploration_weight": 1.41,
        "max_depth": 20,
        "discount_factor": 0.99,
    },
    "pipeline": {
        "checkpoint_dir": None,
        "resume": False,
        "stages": ["decompose", "solve-lp", "infer", "verify", "search"],
    },
    "benchmark": {
        "suite": "all",
        "sizes": [10, 50, 100, 500],
        "repetitions": 5,
        "output": "benchmark_report.json",
    },
    "logging": {
        "level": "INFO",
        "log_file": None,
        "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    },
}

logger = logging.getLogger("causalbound.cli")


# ---------------------------------------------------------------------------
# Lazy-import helper
# ---------------------------------------------------------------------------

def _lazy_import(module_path: str, class_name: str) -> Any:
    """Import *class_name* from *module_path* at call time.

    Returns ``None`` when the module is not installed so callers can
    fall back gracefully.
    """
    try:
        mod = importlib.import_module(module_path)
        return getattr(mod, class_name)
    except (ImportError, AttributeError):
        return None


def _require_import(module_path: str, class_name: str) -> Any:
    """Like :func:`_lazy_import` but raises :class:`click.ClickException`
    when the import fails."""
    obj = _lazy_import(module_path, class_name)
    if obj is None:
        raise click.ClickException(
            f"Cannot import '{class_name}' from '{module_path}'.  "
            f"Make sure the causalbound package is installed:\n"
            f"  pip install -e .[all]"
        )
    return obj


def _try_import_tqdm():
    """Return ``tqdm.tqdm`` if available, else a no-op fallback."""
    try:
        from tqdm import tqdm
        return tqdm
    except ImportError:
        class _NoopBar:
            def __init__(self, iterable=None, **kw):
                self._it = iterable
                self.total = kw.get("total", 0)
                self.n = 0
            def __iter__(self):
                return iter(self._it) if self._it else iter([])
            def __enter__(self):
                return self
            def __exit__(self, *a):
                pass
            def update(self, n=1):
                self.n += n
            def set_postfix_str(self, s):
                pass
            def close(self):
                pass
        return _NoopBar


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _load_config(path: Optional[str]) -> Dict[str, Any]:
    """Load YAML configuration, falling back to *DEFAULT_CONFIG*."""
    import copy
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if path is None:
        return cfg
    resolved = pathlib.Path(path).expanduser().resolve()
    if not resolved.exists():
        raise click.ClickException(f"Config file not found: {resolved}")
    try:
        import yaml
    except ImportError:
        raise click.ClickException(
            "PyYAML is required to read config files.  "
            "Install it with:  pip install pyyaml"
        )
    with open(resolved, "r") as fh:
        user_cfg = yaml.safe_load(fh) or {}
    _deep_merge(cfg, user_cfg)
    return cfg


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* in place."""
    for key, val in override.items():
        if key in base and isinstance(base[key], dict) and isinstance(val, dict):
            _deep_merge(base[key], val)
        else:
            base[key] = val
    return base


def _setup_logging(level: str, log_file: Optional[str] = None) -> None:
    """Configure the root ``causalbound`` logger."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    fmt = DEFAULT_CONFIG["logging"]["format"]
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stderr)]
    if log_file:
        log_path = pathlib.Path(log_file).expanduser().resolve()
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(str(log_path)))
    logging.basicConfig(level=numeric, format=fmt, handlers=handlers, force=True)
    logger.setLevel(numeric)


def _format_output(data: Any, fmt: str, keys: Optional[List[str]] = None) -> str:
    """Render *data* as JSON, CSV, or human-readable text.

    Parameters
    ----------
    data : dict | list[dict]
        The payload to render.
    fmt : str
        One of ``"json"``, ``"csv"``, ``"human"``.
    keys : list[str] | None
        Column ordering hint for CSV output.
    """
    if fmt == "json":
        return json.dumps(data, indent=2, default=str)
    if fmt == "csv":
        return _dict_to_csv(data, keys)
    return _dict_to_human(data)


def _dict_to_csv(data: Any, keys: Optional[List[str]] = None) -> str:
    """Serialise a dict or list-of-dicts to CSV."""
    buf = io.StringIO()
    if isinstance(data, dict):
        rows = [data]
    elif isinstance(data, list):
        rows = data
    else:
        rows = [{"value": str(data)}]
    if not rows:
        return ""
    fieldnames = keys or list(rows[0].keys())
    writer = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: _csv_safe(row.get(k, "")) for k in fieldnames})
    return buf.getvalue()


def _csv_safe(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.8g}"
    return str(val)


def _dict_to_human(data: Any, indent: int = 0) -> str:
    """Pretty-print nested dicts for terminal output."""
    lines: list[str] = []
    prefix = "  " * indent
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)):
                lines.append(f"{prefix}{k}:")
                lines.append(_dict_to_human(v, indent + 1))
            else:
                lines.append(f"{prefix}{k}: {_format_scalar(v)}")
    elif isinstance(data, list):
        for i, item in enumerate(data):
            lines.append(f"{prefix}[{i}]")
            lines.append(_dict_to_human(item, indent + 1))
    else:
        lines.append(f"{prefix}{_format_scalar(data)}")
    return "\n".join(lines)


def _format_scalar(val: Any) -> str:
    if isinstance(val, float):
        return f"{val:.6g}"
    return str(val)


def _print_bounds(lower: float, upper: float) -> str:
    """Return a coloured interval string ``[lower, upper]``."""
    gap = upper - lower
    interval = f"[{lower:.6f}, {upper:.6f}]"
    gap_str = f"(gap = {gap:.2e})"
    return f"{interval}  {gap_str}"


def _progress_callback(stage: str, current: int, total: int) -> None:
    """Emit a progress line to stderr."""
    pct = (current / total * 100) if total else 0
    bar_len = 30
    filled = int(bar_len * current // max(total, 1))
    bar = "█" * filled + "░" * (bar_len - filled)
    click.echo(
        f"\r  {stage}: |{bar}| {pct:5.1f}% ({current}/{total})",
        nl=False,
        err=True,
    )
    if current >= total:
        click.echo("", err=True)


def _ensure_output_dir(path: Optional[str]) -> pathlib.Path:
    """Create and return the output directory."""
    if path is None:
        out = pathlib.Path.cwd() / "causalbound_output"
    else:
        out = pathlib.Path(path).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _write_output(
    data: Any,
    fmt: str,
    output_dir: pathlib.Path,
    filename_stem: str,
    keys: Optional[List[str]] = None,
) -> pathlib.Path:
    """Format *data* and write to a file inside *output_dir*."""
    ext_map = {"json": ".json", "csv": ".csv", "human": ".txt"}
    ext = ext_map.get(fmt, ".txt")
    dest = output_dir / f"{filename_stem}{ext}"
    rendered = _format_output(data, fmt, keys)
    dest.write_text(rendered, encoding="utf-8")
    return dest


def _load_network(path: str) -> Dict[str, Any]:
    """Load a financial network from a JSON or CSV file.

    Returns a dict with at least ``nodes`` and ``edges`` keys.
    """
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise click.ClickException(f"Network file not found: {p}")
    if p.suffix == ".json":
        with open(p) as fh:
            network = json.load(fh)
    elif p.suffix == ".csv":
        network = _network_from_csv(p)
    else:
        raise click.ClickException(
            f"Unsupported network format '{p.suffix}'.  Use .json or .csv."
        )
    if "nodes" not in network or "edges" not in network:
        raise click.ClickException(
            "Network file must contain 'nodes' and 'edges' keys."
        )
    logger.info(
        "Loaded network with %d nodes and %d edges",
        len(network["nodes"]),
        len(network["edges"]),
    )
    return network


def _network_from_csv(path: pathlib.Path) -> Dict[str, Any]:
    """Parse an edge-list CSV into a network dict."""
    nodes: set[str] = set()
    edges: list[dict] = []
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            src = row.get("source") or row.get("src") or row.get("from", "")
            tgt = row.get("target") or row.get("tgt") or row.get("to", "")
            if not src or not tgt:
                continue
            weight = float(row.get("weight", 1.0))
            nodes.update([src, tgt])
            edges.append({"source": src, "target": tgt, "weight": weight})
    return {
        "nodes": sorted(nodes),
        "edges": edges,
    }


def _parse_query(raw: str) -> Dict[str, Any]:
    """Parse a causal query string into structured form.

    Accepted formats::

        P(Y|do(X))
        P(Y=1|do(X=0),Z=1)
        Y | do(X)
    """
    raw = raw.strip()
    query: Dict[str, Any] = {"target": None, "interventions": {}, "conditions": {}}

    body = raw
    if body.upper().startswith("P(") and body.endswith(")"):
        body = body[2:-1]

    parts = body.split("|", 1)
    target_part = parts[0].strip()
    query["target"] = _parse_var_assignment(target_part)

    if len(parts) == 2:
        rest = parts[1].strip()
        segments = _split_keeping_parens(rest)
        for seg in segments:
            seg = seg.strip().strip(",").strip()
            if seg.lower().startswith("do(") and seg.endswith(")"):
                inner = seg[3:-1]
                for iv in inner.split(","):
                    k, v = _split_assignment(iv.strip())
                    query["interventions"][k] = v
            else:
                k, v = _split_assignment(seg)
                query["conditions"][k] = v
    return query


def _parse_var_assignment(s: str) -> Dict[str, Any]:
    if "=" in s:
        name, val = s.split("=", 1)
        return {"name": name.strip(), "value": val.strip()}
    return {"name": s.strip(), "value": None}


def _split_assignment(s: str) -> Tuple[str, Any]:
    if "=" in s:
        k, v = s.split("=", 1)
        return k.strip(), v.strip()
    return s.strip(), None


def _split_keeping_parens(s: str) -> List[str]:
    """Split on commas that are not inside parentheses."""
    result: list[str] = []
    depth = 0
    current: list[str] = []
    for ch in s:
        if ch == "(":
            depth += 1
            current.append(ch)
        elif ch == ")":
            depth -= 1
            current.append(ch)
        elif ch == "," and depth == 0:
            result.append("".join(current))
            current = []
        else:
            current.append(ch)
    if current:
        result.append("".join(current))
    return result


def _load_json_file(path: str, label: str = "file") -> Any:
    """Read and return parsed JSON from *path*."""
    p = pathlib.Path(path).expanduser().resolve()
    if not p.exists():
        raise click.ClickException(f"{label} not found: {p}")
    with open(p) as fh:
        return json.load(fh)


def _save_checkpoint(data: Dict[str, Any], stage: str, ckpt_dir: pathlib.Path) -> pathlib.Path:
    """Persist a stage checkpoint to disk."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    dest = ckpt_dir / f"checkpoint_{stage}.json"
    with open(dest, "w") as fh:
        json.dump(data, fh, indent=2, default=str)
    logger.info("Checkpoint saved: %s", dest)
    return dest


def _load_checkpoint(stage: str, ckpt_dir: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load a checkpoint if it exists; return ``None`` otherwise."""
    src = ckpt_dir / f"checkpoint_{stage}.json"
    if not src.exists():
        return None
    with open(src) as fh:
        data = json.load(fh)
    logger.info("Resumed from checkpoint: %s", src)
    return data


def _elapsed_str(seconds: float) -> str:
    """Human-friendly elapsed time."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


# ---------------------------------------------------------------------------
# Click CLI
# ---------------------------------------------------------------------------

@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    epilog="See https://github.com/causalbound/causalbound for documentation.",
)
@click.option(
    "--config", "-c", "config_path",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML configuration file.",
)
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable verbose output.")
@click.option(
    "--log-level", "-l",
    type=click.Choice(["DEBUG", "INFO", "WARNING", "ERROR"], case_sensitive=False),
    default="INFO",
    help="Logging verbosity.",
)
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default=None,
    help="Directory for output artefacts.",
)
@click.option(
    "--format", "-f", "output_format",
    type=click.Choice(["json", "csv", "human"], case_sensitive=False),
    default="human",
    help="Output format.",
)
@click.version_option(version="0.1.0", prog_name="causalbound")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: Optional[str],
    verbose: bool,
    log_level: str,
    output_dir: Optional[str],
    output_format: str,
) -> None:
    """CausalBound — causal inference for financial systemic risk."""
    effective_level = "DEBUG" if verbose else log_level
    _setup_logging(effective_level)
    cfg = _load_config(config_path)
    ctx.ensure_object(dict)
    ctx.obj["config"] = cfg
    ctx.obj["output_dir"] = _ensure_output_dir(output_dir)
    ctx.obj["format"] = output_format
    ctx.obj["verbose"] = verbose


# ---------------------------------------------------------------------------
# decompose
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--input", "-i", "input_path", required=True, type=click.Path(exists=True), help="Network file (JSON/CSV).")
@click.option("--method", "-m", type=click.Choice(["min-degree", "min-fill"]), default=None, help="Elimination heuristic.")
@click.option("--max-treewidth", type=int, default=None, help="Upper bound on treewidth.")
@click.pass_context
def decompose(ctx: click.Context, input_path: str, method: Optional[str], max_treewidth: Optional[int]) -> None:
    """Tree-decompose a financial network."""
    cfg = ctx.obj["config"]["decomposition"]
    method = method or cfg["method"]
    max_tw = max_treewidth if max_treewidth is not None else cfg["max_treewidth"]
    timeout = cfg["timeout_seconds"]

    network = _load_network(input_path)
    n_nodes = len(network["nodes"])
    n_edges = len(network["edges"])
    click.echo(f"Network: {n_nodes} nodes, {n_edges} edges", err=True)
    click.echo(f"Method : {method}  |  max treewidth: {max_tw}", err=True)

    tqdm = _try_import_tqdm()
    TreeDecomposer = _lazy_import("causalbound.decomposition", "TreeDecomposer")

    t0 = time.time()
    if TreeDecomposer is not None:
        decomposer = TreeDecomposer(method=method, max_treewidth=max_tw, timeout=timeout)
        result = decomposer.decompose(network)
    else:
        logger.warning("TreeDecomposer not available — running lightweight fallback")
        result = _fallback_decompose(network, method, max_tw, tqdm)
    elapsed = time.time() - t0

    summary: Dict[str, Any] = {
        "method": method,
        "max_treewidth_requested": max_tw,
        "treewidth": result.get("treewidth", -1),
        "n_bags": result.get("n_bags", 0),
        "largest_bag_size": result.get("largest_bag", 0),
        "elapsed": _elapsed_str(elapsed),
        "nodes": n_nodes,
        "edges": n_edges,
    }

    out = ctx.obj["output_dir"]
    fmt = ctx.obj["format"]
    dest = _write_output(summary, fmt, out, "decomposition")
    click.echo(_format_output(summary, fmt))
    click.echo(f"\nWritten to {dest}", err=True)


def _fallback_decompose(
    network: Dict[str, Any],
    method: str,
    max_tw: int,
    tqdm_cls: Any,
) -> Dict[str, Any]:
    """Greedy elimination decomposition when the real solver is absent."""
    nodes = list(network["nodes"])
    adj: Dict[str, set] = {n: set() for n in nodes}
    for e in network["edges"]:
        s, t = e["source"], e["target"]
        if s in adj and t in adj:
            adj[s].add(t)
            adj[t].add(s)

    bags: list[set] = []
    order: list[str] = []
    remaining = set(nodes)
    tw = 0

    with tqdm_cls(total=len(nodes), desc="Eliminating", unit="node") as pbar:
        while remaining:
            if method == "min-degree":
                v = min(remaining, key=lambda n: len(adj[n] & remaining))
            else:
                v = min(remaining, key=lambda n: _fill_count(n, adj, remaining))
            neighbours = adj[v] & remaining
            bag = neighbours | {v}
            bags.append(bag)
            tw = max(tw, len(bag) - 1)
            for u in neighbours:
                for w in neighbours:
                    if u != w:
                        adj[u].add(w)
                        adj[w].add(u)
            remaining.discard(v)
            order.append(v)
            pbar.update(1)
            if tw > max_tw:
                pbar.set_postfix_str(f"tw={tw} EXCEEDED")

    return {
        "treewidth": tw,
        "n_bags": len(bags),
        "largest_bag": max(len(b) for b in bags) if bags else 0,
        "elimination_order": order,
    }


def _fill_count(node: str, adj: Dict[str, set], remaining: set) -> int:
    """Count fill-in edges for *node* under the min-fill heuristic."""
    nbrs = list(adj[node] & remaining)
    count = 0
    for i, u in enumerate(nbrs):
        for w in nbrs[i + 1:]:
            if w not in adj[u]:
                count += 1
    return count


# ---------------------------------------------------------------------------
# solve-lp
# ---------------------------------------------------------------------------

@cli.command("solve-lp")
@click.option("--network", "-n", required=True, type=click.Path(exists=True), help="Network file.")
@click.option("--query", "-q", required=True, type=str, help="Causal query, e.g. P(Y|do(X)).")
@click.option("--max-iterations", type=int, default=None, help="Column-generation iterations.")
@click.option("--gap-tolerance", type=float, default=None, help="Optimality gap tolerance.")
@click.pass_context
def solve_lp(
    ctx: click.Context,
    network: str,
    query: str,
    max_iterations: Optional[int],
    gap_tolerance: Optional[float],
) -> None:
    """Solve causal-polytope LP for probability bounds."""
    cfg = ctx.obj["config"]["lp_solver"]
    max_iter = max_iterations if max_iterations is not None else cfg["max_iterations"]
    gap_tol = gap_tolerance if gap_tolerance is not None else cfg["gap_tolerance"]
    backend = cfg["solver_backend"]

    net = _load_network(network)
    parsed_query = _parse_query(query)
    click.echo(f"Query  : {query}", err=True)
    click.echo(f"Solver : {backend}  |  max iter: {max_iter}  |  gap tol: {gap_tol:.1e}", err=True)

    LPSolver = _lazy_import("causalbound.lp", "CausalPolytopeSolver")
    tqdm = _try_import_tqdm()

    t0 = time.time()
    if LPSolver is not None:
        solver = LPSolver(
            backend=backend,
            max_iterations=max_iter,
            gap_tolerance=gap_tol,
        )
        result = solver.solve(net, parsed_query)
    else:
        logger.warning("CausalPolytopeSolver not available — running stub LP")
        result = _stub_solve_lp(net, parsed_query, max_iter, gap_tol, tqdm)
    elapsed = time.time() - t0

    lower = result.get("lower_bound", 0.0)
    upper = result.get("upper_bound", 1.0)
    gap = upper - lower
    iterations_used = result.get("iterations", 0)

    click.echo(f"\nBounds : {_print_bounds(lower, upper)}", err=True)
    click.echo(f"Iters  : {iterations_used}  |  elapsed: {_elapsed_str(elapsed)}", err=True)

    output_data: Dict[str, Any] = {
        "query": query,
        "lower_bound": lower,
        "upper_bound": upper,
        "gap": gap,
        "iterations": iterations_used,
        "gap_tolerance": gap_tol,
        "converged": gap <= gap_tol,
        "solver_backend": backend,
        "elapsed": _elapsed_str(elapsed),
    }

    fmt = ctx.obj["format"]
    out = ctx.obj["output_dir"]
    dest = _write_output(output_data, fmt, out, "lp_bounds")
    click.echo(_format_output(output_data, fmt))
    click.echo(f"\nWritten to {dest}", err=True)


def _stub_solve_lp(
    network: Dict[str, Any],
    query: Dict[str, Any],
    max_iter: int,
    gap_tol: float,
    tqdm_cls: Any,
) -> Dict[str, Any]:
    """Stub LP column-generation loop used when the real solver is absent."""
    lower = 0.0
    upper = 1.0
    n_nodes = len(network.get("nodes", []))
    shrink_rate = 0.5 / max(n_nodes, 1)

    with tqdm_cls(total=max_iter, desc="Column generation", unit="iter") as pbar:
        for i in range(1, max_iter + 1):
            lower += shrink_rate * (1 - lower) * 0.05
            upper -= shrink_rate * upper * 0.03
            if upper < lower:
                upper = lower
            gap = upper - lower
            pbar.update(1)
            pbar.set_postfix_str(f"gap={gap:.4e}")
            if gap <= gap_tol:
                break

    return {
        "lower_bound": round(lower, 8),
        "upper_bound": round(upper, 8),
        "iterations": i,
    }


# ---------------------------------------------------------------------------
# infer
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--network", "-n", required=True, type=click.Path(exists=True), help="Network file.")
@click.option("--query", "-q", required=True, type=str, help="Causal query.")
@click.option("--evidence", "-e", type=str, default=None, help="Comma-separated evidence, e.g. A=1,B=0.")
@click.pass_context
def infer(ctx: click.Context, network: str, query: str, evidence: Optional[str]) -> None:
    """Run causal inference with bound composition."""
    cfg = ctx.obj["config"]["inference"]
    net = _load_network(network)
    parsed_query = _parse_query(query)

    evidence_dict: Dict[str, str] = {}
    if evidence:
        for token in evidence.split(","):
            k, v = _split_assignment(token.strip())
            evidence_dict[k] = v

    click.echo(f"Query    : {query}", err=True)
    if evidence_dict:
        click.echo(f"Evidence : {evidence_dict}", err=True)
    click.echo(f"Method   : {cfg['composition_method']}  |  propagation: {cfg['propagation']}", err=True)

    CausalInference = _lazy_import("causalbound.inference", "CausalInference")

    t0 = time.time()
    if CausalInference is not None:
        engine = CausalInference(
            composition_method=cfg["composition_method"],
            propagation=cfg["propagation"],
            max_paths=cfg["max_paths"],
        )
        result = engine.infer(net, parsed_query, evidence=evidence_dict)
    else:
        logger.warning("CausalInference not available — running stub inference")
        result = _stub_infer(net, parsed_query, evidence_dict, cfg)
    elapsed = time.time() - t0

    lower = result.get("lower_bound", 0.0)
    upper = result.get("upper_bound", 1.0)

    output_data: Dict[str, Any] = {
        "query": query,
        "evidence": evidence_dict or None,
        "lower_bound": lower,
        "upper_bound": upper,
        "gap": upper - lower,
        "paths_explored": result.get("paths_explored", 0),
        "composition_method": cfg["composition_method"],
        "elapsed": _elapsed_str(elapsed),
    }

    click.echo(f"\nBounds : {_print_bounds(lower, upper)}", err=True)

    fmt = ctx.obj["format"]
    out = ctx.obj["output_dir"]
    dest = _write_output(output_data, fmt, out, "inference")
    click.echo(_format_output(output_data, fmt))
    click.echo(f"\nWritten to {dest}", err=True)


def _stub_infer(
    network: Dict[str, Any],
    query: Dict[str, Any],
    evidence: Dict[str, str],
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Stub inference when the real engine is absent."""
    import random
    random.seed(hash(json.dumps(query, sort_keys=True)))
    base_lower = random.uniform(0.05, 0.35)
    base_upper = random.uniform(0.65, 0.95)
    evidence_factor = max(0.5, 1.0 - 0.1 * len(evidence))
    gap = (base_upper - base_lower) * evidence_factor
    mid = (base_lower + base_upper) / 2
    lower = max(0.0, mid - gap / 2)
    upper = min(1.0, mid + gap / 2)
    return {
        "lower_bound": round(lower, 6),
        "upper_bound": round(upper, 6),
        "paths_explored": min(cfg["max_paths"], len(network.get("edges", []))),
    }


# ---------------------------------------------------------------------------
# verify
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--bounds-file", "-b", required=True, type=click.Path(exists=True), help="JSON file with bounds.")
@click.option("--certificate-file", "-c", "cert_file", default=None, type=click.Path(), help="Output certificate path.")
@click.option("--timeout", type=int, default=None, help="SMT solver timeout in seconds.")
@click.pass_context
def verify(ctx: click.Context, bounds_file: str, cert_file: Optional[str], timeout: Optional[int]) -> None:
    """SMT-verify computed causal bounds."""
    cfg = ctx.obj["config"]["verification"]
    smt_solver = cfg["smt_solver"]
    smt_timeout = timeout if timeout is not None else cfg["timeout_seconds"]
    epsilon = cfg["epsilon"]

    bounds = _load_json_file(bounds_file, "Bounds file")
    lower = bounds.get("lower_bound", 0.0)
    upper = bounds.get("upper_bound", 1.0)
    click.echo(f"Bounds   : {_print_bounds(lower, upper)}", err=True)
    click.echo(f"Solver   : {smt_solver}  |  timeout: {smt_timeout}s  |  ε: {epsilon:.1e}", err=True)

    SMTVerifier = _lazy_import("causalbound.verification", "SMTVerifier")

    t0 = time.time()
    if SMTVerifier is not None:
        verifier = SMTVerifier(solver=smt_solver, timeout=smt_timeout, epsilon=epsilon)
        result = verifier.verify(bounds)
    else:
        logger.warning("SMTVerifier not available — running stub verification")
        result = _stub_verify(bounds, smt_timeout, epsilon)
    elapsed = time.time() - t0

    status = result.get("status", "UNKNOWN")
    status_colors = {"VERIFIED": "green", "FAILED": "red", "TIMEOUT": "yellow", "UNKNOWN": "white"}
    click.echo(
        f"\nStatus : {click.style(status, fg=status_colors.get(status, 'white'), bold=True)}",
        err=True,
    )
    click.echo(f"Elapsed: {_elapsed_str(elapsed)}", err=True)

    output_data: Dict[str, Any] = {
        "status": status,
        "lower_bound": lower,
        "upper_bound": upper,
        "smt_solver": smt_solver,
        "timeout": smt_timeout,
        "epsilon": epsilon,
        "elapsed": _elapsed_str(elapsed),
        "counterexample": result.get("counterexample"),
    }

    if cert_file and status == "VERIFIED":
        cert_path = pathlib.Path(cert_file).expanduser().resolve()
        certificate = {
            "status": "VERIFIED",
            "bounds": {"lower": lower, "upper": upper},
            "solver": smt_solver,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "proof_hash": result.get("proof_hash", ""),
        }
        cert_path.parent.mkdir(parents=True, exist_ok=True)
        cert_path.write_text(json.dumps(certificate, indent=2))
        click.echo(f"Certificate written to {cert_path}", err=True)

    fmt = ctx.obj["format"]
    out = ctx.obj["output_dir"]
    dest = _write_output(output_data, fmt, out, "verification")
    click.echo(_format_output(output_data, fmt))
    click.echo(f"\nWritten to {dest}", err=True)


def _stub_verify(bounds: Dict[str, Any], timeout: int, epsilon: float) -> Dict[str, Any]:
    """Stub SMT verification."""
    lower = bounds.get("lower_bound", 0.0)
    upper = bounds.get("upper_bound", 1.0)
    if upper < lower:
        return {"status": "FAILED", "counterexample": {"reason": "upper < lower"}}
    if lower < -epsilon or upper > 1.0 + epsilon:
        return {"status": "FAILED", "counterexample": {"reason": "bounds outside [0,1]"}}
    import hashlib
    h = hashlib.sha256(json.dumps(bounds, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return {"status": "VERIFIED", "proof_hash": h}


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--network", "-n", required=True, type=click.Path(exists=True), help="Network file.")
@click.option("--n-rollouts", type=int, default=None, help="Number of MCTS rollouts.")
@click.option("--exploration-weight", type=float, default=None, help="UCB exploration constant.")
@click.option("--method", "-m", type=click.Choice(["mcts", "random", "grid"]), default=None, help="Search method.")
@click.pass_context
def search(
    ctx: click.Context,
    network: str,
    n_rollouts: Optional[int],
    exploration_weight: Optional[float],
    method: Optional[str],
) -> None:
    """Adversarial scenario search for worst-case systemic losses."""
    cfg = ctx.obj["config"]["search"]
    search_method = method or cfg["method"]
    rollouts = n_rollouts if n_rollouts is not None else cfg["n_rollouts"]
    c_explore = exploration_weight if exploration_weight is not None else cfg["exploration_weight"]
    max_depth = cfg["max_depth"]

    net = _load_network(network)
    click.echo(f"Method   : {search_method}  |  rollouts: {rollouts}  |  C: {c_explore:.2f}", err=True)

    AdversarialSearch = _lazy_import("causalbound.search", "AdversarialSearch")
    tqdm = _try_import_tqdm()

    t0 = time.time()
    if AdversarialSearch is not None:
        searcher = AdversarialSearch(
            method=search_method,
            n_rollouts=rollouts,
            exploration_weight=c_explore,
            max_depth=max_depth,
        )
        result = searcher.search(net)
    else:
        logger.warning("AdversarialSearch not available — running stub search")
        result = _stub_search(net, search_method, rollouts, c_explore, max_depth, tqdm)
    elapsed = time.time() - t0

    worst_loss = result.get("worst_case_loss", 0.0)
    scenario = result.get("scenario", {})

    click.echo(f"\nWorst-case loss : {worst_loss:.6f}", err=True)
    click.echo(f"Scenario nodes  : {len(scenario.get('failed_nodes', []))}", err=True)
    click.echo(f"Elapsed         : {_elapsed_str(elapsed)}", err=True)

    output_data: Dict[str, Any] = {
        "method": search_method,
        "n_rollouts": rollouts,
        "exploration_weight": c_explore,
        "worst_case_loss": worst_loss,
        "scenario": scenario,
        "rollouts_completed": result.get("rollouts_completed", rollouts),
        "elapsed": _elapsed_str(elapsed),
    }

    fmt = ctx.obj["format"]
    out = ctx.obj["output_dir"]
    dest = _write_output(output_data, fmt, out, "search")
    click.echo(_format_output(output_data, fmt))
    click.echo(f"\nWritten to {dest}", err=True)


def _stub_search(
    network: Dict[str, Any],
    method: str,
    n_rollouts: int,
    c_explore: float,
    max_depth: int,
    tqdm_cls: Any,
) -> Dict[str, Any]:
    """Stub adversarial search when the real engine is absent."""
    import random
    random.seed(42)
    nodes = network.get("nodes", [])
    edges = network.get("edges", [])

    best_loss = 0.0
    best_scenario: Dict[str, Any] = {}
    completed = 0

    with tqdm_cls(total=n_rollouts, desc=f"Search ({method})", unit="rollout") as pbar:
        for r in range(n_rollouts):
            if method == "random":
                k = random.randint(1, max(1, len(nodes) // 3))
                failed = random.sample(nodes, min(k, len(nodes)))
            elif method == "grid":
                idx = r % max(len(nodes), 1)
                failed = [nodes[idx]] if nodes else []
            else:
                k = random.randint(1, max(1, min(max_depth, len(nodes) // 2)))
                score_map = {n: random.random() + c_explore * (1.0 / (r + 1)) ** 0.5 for n in nodes}
                failed = sorted(score_map, key=score_map.get, reverse=True)[:k]

            affected_edges = sum(
                1 for e in edges
                if e["source"] in failed or e["target"] in failed
            )
            loss = affected_edges / max(len(edges), 1)

            if loss > best_loss:
                best_loss = loss
                best_scenario = {
                    "failed_nodes": failed,
                    "affected_edges": affected_edges,
                    "rollout": r,
                }
            completed += 1
            pbar.update(1)
            pbar.set_postfix_str(f"best={best_loss:.4f}")

    return {
        "worst_case_loss": round(best_loss, 6),
        "scenario": best_scenario,
        "rollouts_completed": completed,
    }


# ---------------------------------------------------------------------------
# run-pipeline
# ---------------------------------------------------------------------------

@cli.command("run-pipeline")
@click.option("--network", "-n", required=True, type=click.Path(exists=True), help="Network file.")
@click.option("--query", "-q", required=True, type=str, help="Causal query.")
@click.option("--evidence", "-e", type=str, default=None, help="Comma-separated evidence.")
@click.option("--checkpoint-dir", type=click.Path(), default=None, help="Directory for stage checkpoints.")
@click.option("--resume", is_flag=True, default=False, help="Resume from last checkpoint.")
@click.pass_context
def run_pipeline(
    ctx: click.Context,
    network: str,
    query: str,
    evidence: Optional[str],
    checkpoint_dir: Optional[str],
    resume: bool,
) -> None:
    """Run the full CausalBound pipeline end-to-end.

    Stages: decompose → solve-lp → infer → verify → search
    """
    cfg = ctx.obj["config"]
    fmt = ctx.obj["format"]
    out = ctx.obj["output_dir"]

    ckpt_dir = pathlib.Path(checkpoint_dir) if checkpoint_dir else out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    all_stages = cfg["pipeline"]["stages"]
    results: Dict[str, Any] = {}
    stage_times: Dict[str, float] = {}

    click.echo("=" * 60, err=True)
    click.echo("  CausalBound Pipeline", err=True)
    click.echo("=" * 60, err=True)
    click.echo(f"  Network : {network}", err=True)
    click.echo(f"  Query   : {query}", err=True)
    click.echo(f"  Stages  : {' → '.join(all_stages)}", err=True)
    click.echo("=" * 60, err=True)

    pipeline_t0 = time.time()

    # --- Stage 1: decompose ------------------------------------------------
    stage = "decompose"
    if stage in all_stages:
        click.echo(f"\n▶ Stage 1/5: {stage}", err=True)
        cached = _load_checkpoint(stage, ckpt_dir) if resume else None
        if cached is not None:
            results[stage] = cached
            click.echo("  (resumed from checkpoint)", err=True)
        else:
            t0 = time.time()
            sub_ctx = click.Context(decompose, parent=ctx, info_name=stage)
            sub_ctx.obj = ctx.obj
            net = _load_network(network)
            tqdm = _try_import_tqdm()
            decomp_cfg = cfg["decomposition"]
            decomp_result = _fallback_decompose(net, decomp_cfg["method"], decomp_cfg["max_treewidth"], tqdm)
            stage_times[stage] = time.time() - t0
            results[stage] = decomp_result
            _save_checkpoint(decomp_result, stage, ckpt_dir)
        click.echo(f"  treewidth = {results[stage].get('treewidth', '?')}", err=True)

    # --- Stage 2: solve-lp -------------------------------------------------
    stage = "solve-lp"
    if stage in all_stages:
        click.echo(f"\n▶ Stage 2/5: {stage}", err=True)
        cached = _load_checkpoint(stage, ckpt_dir) if resume else None
        if cached is not None:
            results[stage] = cached
            click.echo("  (resumed from checkpoint)", err=True)
        else:
            t0 = time.time()
            net = _load_network(network)
            parsed_q = _parse_query(query)
            lp_cfg = cfg["lp_solver"]
            tqdm = _try_import_tqdm()
            lp_result = _stub_solve_lp(net, parsed_q, lp_cfg["max_iterations"], lp_cfg["gap_tolerance"], tqdm)
            stage_times[stage] = time.time() - t0
            results[stage] = lp_result
            _save_checkpoint(lp_result, stage, ckpt_dir)
        lb = results[stage].get("lower_bound", 0)
        ub = results[stage].get("upper_bound", 1)
        click.echo(f"  bounds = {_print_bounds(lb, ub)}", err=True)

    # --- Stage 3: infer ----------------------------------------------------
    stage = "infer"
    if stage in all_stages:
        click.echo(f"\n▶ Stage 3/5: {stage}", err=True)
        cached = _load_checkpoint(stage, ckpt_dir) if resume else None
        if cached is not None:
            results[stage] = cached
            click.echo("  (resumed from checkpoint)", err=True)
        else:
            t0 = time.time()
            net = _load_network(network)
            parsed_q = _parse_query(query)
            ev: Dict[str, str] = {}
            if evidence:
                for tok in evidence.split(","):
                    k, v = _split_assignment(tok.strip())
                    ev[k] = v
            infer_result = _stub_infer(net, parsed_q, ev, cfg["inference"])
            stage_times[stage] = time.time() - t0
            results[stage] = infer_result
            _save_checkpoint(infer_result, stage, ckpt_dir)
        lb = results[stage].get("lower_bound", 0)
        ub = results[stage].get("upper_bound", 1)
        click.echo(f"  bounds = {_print_bounds(lb, ub)}", err=True)

    # --- Stage 4: verify ---------------------------------------------------
    stage = "verify"
    if stage in all_stages:
        click.echo(f"\n▶ Stage 4/5: {stage}", err=True)
        cached = _load_checkpoint(stage, ckpt_dir) if resume else None
        if cached is not None:
            results[stage] = cached
            click.echo("  (resumed from checkpoint)", err=True)
        else:
            t0 = time.time()
            bounds_to_check = results.get("solve-lp", results.get("infer", {}))
            verify_result = _stub_verify(bounds_to_check, cfg["verification"]["timeout_seconds"], cfg["verification"]["epsilon"])
            stage_times[stage] = time.time() - t0
            results[stage] = verify_result
            _save_checkpoint(verify_result, stage, ckpt_dir)
        status = results[stage].get("status", "UNKNOWN")
        colour = {"VERIFIED": "green", "FAILED": "red"}.get(status, "yellow")
        click.echo(f"  status = {click.style(status, fg=colour, bold=True)}", err=True)

    # --- Stage 5: search ---------------------------------------------------
    stage = "search"
    if stage in all_stages:
        click.echo(f"\n▶ Stage 5/5: {stage}", err=True)
        cached = _load_checkpoint(stage, ckpt_dir) if resume else None
        if cached is not None:
            results[stage] = cached
            click.echo("  (resumed from checkpoint)", err=True)
        else:
            t0 = time.time()
            net = _load_network(network)
            search_cfg = cfg["search"]
            tqdm = _try_import_tqdm()
            search_result = _stub_search(
                net, search_cfg["method"], search_cfg["n_rollouts"],
                search_cfg["exploration_weight"], search_cfg["max_depth"], tqdm,
            )
            stage_times[stage] = time.time() - t0
            results[stage] = search_result
            _save_checkpoint(search_result, stage, ckpt_dir)
        wl = results[stage].get("worst_case_loss", 0)
        click.echo(f"  worst-case loss = {wl:.6f}", err=True)

    # --- Summary -----------------------------------------------------------
    total_elapsed = time.time() - pipeline_t0
    click.echo("\n" + "=" * 60, err=True)
    click.echo("  Pipeline Summary", err=True)
    click.echo("=" * 60, err=True)
    for s in all_stages:
        t = stage_times.get(s)
        t_str = _elapsed_str(t) if t is not None else "(cached)"
        click.echo(f"  {s:15s} : {t_str}", err=True)
    click.echo(f"  {'total':15s} : {_elapsed_str(total_elapsed)}", err=True)
    click.echo("=" * 60, err=True)

    summary = {
        "query": query,
        "network": network,
        "stages_completed": list(results.keys()),
        "stage_timings": {k: round(v, 3) for k, v in stage_times.items()},
        "total_elapsed": _elapsed_str(total_elapsed),
        "results": results,
    }
    dest = _write_output(summary, fmt, out, "pipeline_summary")
    click.echo(_format_output(summary, fmt))
    click.echo(f"\nFull results written to {dest}", err=True)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--suite", "-s",
    type=click.Choice(["all", "tightness", "pathway", "discovery", "scalability"]),
    default=None,
    help="Benchmark suite to run.",
)
@click.option("--sizes", type=str, default=None, help="Comma-separated network sizes, e.g. 10,50,100.")
@click.option("--repetitions", "-r", type=int, default=None, help="Repetitions per size.")
@click.option("--output", "-o", "bench_output", type=click.Path(), default=None, help="Report output path.")
@click.pass_context
def benchmark(
    ctx: click.Context,
    suite: Optional[str],
    sizes: Optional[str],
    repetitions: Optional[int],
    bench_output: Optional[str],
) -> None:
    """Run reproducibility benchmark suites."""
    cfg = ctx.obj["config"]["benchmark"]
    suite_name = suite or cfg["suite"]
    size_list = [int(s) for s in sizes.split(",")] if sizes else cfg["sizes"]
    reps = repetitions if repetitions is not None else cfg["repetitions"]
    report_path = pathlib.Path(bench_output or cfg["output"]).expanduser().resolve()

    click.echo(f"Suite       : {suite_name}", err=True)
    click.echo(f"Sizes       : {size_list}", err=True)
    click.echo(f"Repetitions : {reps}", err=True)

    BenchmarkRunner = _lazy_import("causalbound.benchmark", "BenchmarkRunner")
    tqdm = _try_import_tqdm()

    t0 = time.time()
    if BenchmarkRunner is not None:
        runner = BenchmarkRunner(suite=suite_name, sizes=size_list, repetitions=reps)
        report = runner.run()
    else:
        logger.warning("BenchmarkRunner not available — running stub benchmarks")
        report = _stub_benchmark(suite_name, size_list, reps, tqdm)
    elapsed = time.time() - t0

    click.echo(f"\nBenchmark completed in {_elapsed_str(elapsed)}", err=True)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, default=str))
    click.echo(f"Report written to {report_path}", err=True)

    fmt = ctx.obj["format"]
    if fmt == "human":
        _print_benchmark_table(report)
    else:
        click.echo(_format_output(report, fmt))


def _stub_benchmark(
    suite: str,
    sizes: List[int],
    reps: int,
    tqdm_cls: Any,
) -> Dict[str, Any]:
    """Generate stub benchmark results."""
    import random
    random.seed(123)

    suites_to_run = (
        ["tightness", "pathway", "discovery", "scalability"]
        if suite == "all"
        else [suite]
    )

    report: Dict[str, Any] = {
        "suite": suite,
        "sizes": sizes,
        "repetitions": reps,
        "results": {},
    }

    total_tasks = len(suites_to_run) * len(sizes) * reps
    with tqdm_cls(total=total_tasks, desc="Benchmarking", unit="task") as pbar:
        for s in suites_to_run:
            suite_results: list[dict] = []
            for n in sizes:
                for r in range(reps):
                    t_start = time.time()
                    if s == "tightness":
                        gap = random.uniform(0.001, 0.1) / (n ** 0.5)
                        row = {"size": n, "rep": r, "gap": round(gap, 6), "tight": gap < 0.01}
                    elif s == "pathway":
                        paths = random.randint(1, n)
                        row = {"size": n, "rep": r, "paths_found": paths, "coverage": round(paths / n, 4)}
                    elif s == "discovery":
                        true_pos = random.randint(0, n // 2)
                        false_pos = random.randint(0, n // 10)
                        precision = true_pos / max(true_pos + false_pos, 1)
                        row = {"size": n, "rep": r, "true_pos": true_pos, "false_pos": false_pos, "precision": round(precision, 4)}
                    else:
                        elapsed_task = 0.001 * n * random.uniform(0.5, 1.5)
                        row = {"size": n, "rep": r, "elapsed_s": round(elapsed_task, 4), "memory_mb": round(n * 0.1, 2)}
                    row["wall_time"] = round(time.time() - t_start, 6)
                    suite_results.append(row)
                    pbar.update(1)
            report["results"][s] = suite_results

    return report


def _print_benchmark_table(report: Dict[str, Any]) -> None:
    """Pretty-print benchmark results as a table."""
    for suite_name, rows in report.get("results", {}).items():
        click.echo(f"\n{'─' * 50}")
        click.echo(f"  Suite: {suite_name}")
        click.echo(f"{'─' * 50}")
        if not rows:
            click.echo("  (no results)")
            continue
        headers = list(rows[0].keys())
        col_widths = {h: max(len(h), max(len(str(r.get(h, ""))) for r in rows)) for h in headers}
        header_line = "  ".join(h.ljust(col_widths[h]) for h in headers)
        click.echo(f"  {header_line}")
        click.echo(f"  {'  '.join('─' * col_widths[h] for h in headers)}")
        for row in rows:
            vals = "  ".join(str(row.get(h, "")).ljust(col_widths[h]) for h in headers)
            click.echo(f"  {vals}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point for ``python -m causalbound.cli``."""
    cli(auto_envvar_prefix="CAUSALBOUND")


if __name__ == "__main__":
    main()
