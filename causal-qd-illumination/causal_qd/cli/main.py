"""Click-based CLI entry point for CausalQD experiments.

Provides the following commands:
  - ``causal-qd run``        : Run a CausalQD experiment
  - ``causal-qd evaluate``   : Evaluate results against ground truth
  - ``causal-qd visualize``  : Generate plots from results
  - ``causal-qd benchmark``  : Run standard benchmark experiments
  - ``causal-qd certificate``: Compute robustness certificates
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
import numpy as np

from causal_qd.config.config import CausalQDConfig

logger = logging.getLogger("causal_qd")


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the CLI."""
    level = logging.DEBUG if verbose else logging.INFO
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    ))
    logging.getLogger("causal_qd").setLevel(level)
    logging.getLogger("causal_qd").addHandler(handler)


def _load_data(data_path: str) -> np.ndarray:
    """Load data from CSV, NPZ, or Parquet, handling headers."""
    path = Path(data_path)
    if not path.exists():
        raise click.ClickException(f"Data file not found: {data_path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        from causal_qd.data.loader import DataLoader
        return DataLoader.load_parquet(data_path)
    elif suffix in (".npy", ".npz"):
        from causal_qd.data.loader import DataLoader
        return DataLoader.load_numpy(data_path)
    else:
        # Default: CSV
        try:
            data = np.loadtxt(data_path, delimiter=",", skiprows=1)
        except ValueError:
            data = np.loadtxt(data_path, delimiter=",")
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        return data


def _load_ground_truth(gt_path: str) -> np.ndarray:
    """Load a ground-truth DAG from a DOT file or adjacency-matrix CSV."""
    path = Path(gt_path)
    if not path.exists():
        raise click.ClickException(f"Ground-truth file not found: {gt_path}")

    suffix = path.suffix.lower()
    if suffix == ".dot":
        try:
            import networkx as nx  # type: ignore[import-untyped]
            from networkx.drawing.nx_pydot import read_dot  # type: ignore[import-untyped]
        except ImportError:
            raise click.ClickException(
                "networkx and pydot are required for DOT file support. "
                "Install with: pip install networkx pydot"
            )
        G = read_dot(gt_path)
        nodes = sorted(G.nodes())
        n = len(nodes)
        node_idx = {node: i for i, node in enumerate(nodes)}
        adj = np.zeros((n, n), dtype=np.int8)
        for u, v in G.edges():
            adj[node_idx[u], node_idx[v]] = 1
        return adj
    else:
        # CSV adjacency matrix
        try:
            adj = np.loadtxt(gt_path, delimiter=",", skiprows=1).astype(np.int8)
        except ValueError:
            adj = np.loadtxt(gt_path, delimiter=",").astype(np.int8)
        return adj


def _format_output(results: dict, fmt: str) -> str:
    """Format result dict as text, json, or csv."""
    if fmt == "json":
        return json.dumps(results, indent=2, default=str)
    elif fmt == "csv":
        keys = list(results.keys())
        # Only include scalar values in CSV
        scalar_keys = [k for k in keys if isinstance(results[k], (int, float, str, type(None)))]
        header = ",".join(scalar_keys)
        values = ",".join(str(results[k]) for k in scalar_keys)
        return f"{header}\n{values}"
    else:
        # text
        lines = ["=" * 40]
        for key, val in results.items():
            if isinstance(val, (list, dict)):
                continue
            if isinstance(val, float):
                lines.append(f"  {key:20s}: {val:.4f}")
            else:
                lines.append(f"  {key:20s}: {val}")
        lines.append("=" * 40)
        return "\n".join(lines)


# ======================================================================
# Main CLI group
# ======================================================================


@click.group()
@click.option("--verbose/--quiet", default=False, help="Enable verbose logging.")
@click.version_option(version="0.1.0", prog_name="causal-qd")
def main(verbose: bool) -> None:
    """CausalQD — Quality-Diversity for Causal Discovery.

    A framework for illuminating the space of causal models using
    MAP-Elites with certificate-guided quality metrics.
    """
    _setup_logging(verbose)


# ======================================================================
# Run command
# ======================================================================


@main.command()
@click.option("--config", "config_path", type=click.Path(exists=True), default=None,
              help="Path to YAML/JSON config file.")
@click.option("--data", "data_path", type=click.Path(exists=True), required=True,
              help="Path to observational data (CSV, NPZ, or Parquet).")
@click.option("--n-vars", type=int, default=None,
              help="Number of variables (auto-detected from data if omitted).")
@click.option("--n-generations", "n_iterations", type=int, default=None,
              help="Number of MAP-Elites generations.")
@click.option("--batch-size", type=int, default=None,
              help="Offspring per generation.")
@click.option("--archive-type", type=click.Choice(["grid", "cvt"]), default=None,
              help="Archive type.")
@click.option("--score-type", type=click.Choice(["bic", "bdeu", "bge"]), default=None,
              help="Scoring function type.")
@click.option("--output", "output_dir", type=click.Path(), default="results",
              help="Output directory.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--n-workers", type=int, default=1, help="Number of parallel workers.")
@click.option("--n-bootstrap", type=int, default=None, help="Number of bootstrap samples.")
@click.option("--output-format", type=click.Choice(["text", "json", "csv"]),
              default="text", help="Output format for results summary.")
@click.option("--ground-truth", "ground_truth_path", type=click.Path(exists=True),
              default=None,
              help="Path to ground-truth DAG (DOT file or adjacency-matrix CSV) "
                   "for automatic evaluation during the run.")
def run(
    config_path: Optional[str],
    data_path: str,
    n_vars: Optional[int],
    n_iterations: Optional[int],
    batch_size: Optional[int],
    archive_type: Optional[str],
    score_type: Optional[str],
    output_dir: str,
    seed: int,
    n_workers: int,
    n_bootstrap: Optional[int],
    output_format: str,
    ground_truth_path: Optional[str],
) -> None:
    """Run a CausalQD MAP-Elites experiment.

    Loads observational data from CSV/NPZ/Parquet, configures the archive and
    scoring, then runs the MAP-Elites loop.  Results are saved to the output
    directory.  Optionally evaluates against a ground-truth DAG.
    """
    t0 = time.time()

    # Load config (supports YAML and JSON)
    if config_path is not None:
        p = Path(config_path)
        if p.suffix.lower() == ".json":
            cfg = CausalQDConfig.from_json(config_path)
        else:
            cfg = CausalQDConfig.from_yaml(config_path)
    else:
        cfg = CausalQDConfig()

    # Override config with CLI options
    cfg.data_path = data_path
    cfg.experiment.seed = seed
    cfg.experiment.output_dir = output_dir
    cfg.experiment.n_workers = n_workers

    if n_iterations is not None:
        cfg.experiment.n_iterations = n_iterations
    if batch_size is not None:
        cfg.experiment.batch_size = batch_size
    if archive_type is not None:
        cfg.archive.archive_type = archive_type
    if score_type is not None:
        cfg.score.score_type = score_type
    if n_bootstrap is not None:
        cfg.experiment.n_bootstrap = n_bootstrap

    # Load data
    click.echo(f"Loading data from {data_path}...")
    data = _load_data(data_path)
    n_samples, n_nodes = data.shape
    cfg.n_nodes = n_vars if n_vars is not None else n_nodes
    click.echo(f"  Loaded {n_samples} samples × {n_nodes} variables")

    # Run experiment
    click.echo(f"Starting CausalQD (generations={cfg.experiment.n_iterations}, "
               f"batch={cfg.experiment.batch_size}, seed={seed})...")

    from causal_qd.cli.experiment_runner import ExperimentRunner

    with click.progressbar(length=cfg.experiment.n_iterations,
                           label="MAP-Elites") as bar:
        runner = ExperimentRunner(cfg, progress_callback=lambda: bar.update(1))
        results = runner.run(data)

    elapsed = time.time() - t0
    results["elapsed_seconds"] = elapsed

    # Evaluate against ground truth if provided
    if ground_truth_path is not None:
        true_adj = _load_ground_truth(ground_truth_path)
        best_adj_raw = results.get("best_adjacency", [])
        if best_adj_raw:
            best_adj = np.array(best_adj_raw, dtype=np.int8)
            from causal_qd.metrics.structural import SHD, F1

            results["shd"] = int(SHD.compute(best_adj, true_adj))
            f1_m = F1()
            results["f1"] = float(f1_m.compute(best_adj, true_adj))
            results["precision"] = float(f1_m.precision())
            results["recall"] = float(f1_m.recall())
            click.echo("Ground-truth evaluation:")
            click.echo(f"  SHD={results['shd']}  F1={results['f1']:.4f}  "
                       f"Precision={results['precision']:.4f}  Recall={results['recall']:.4f}")

    # Save results
    from causal_qd.cli.result_reporter import ResultReporter

    reporter = ResultReporter(output_dir)
    reporter.report(results)

    if output_format == "text":
        click.echo(reporter.summary(results))
    else:
        click.echo(_format_output(results, output_format))
    click.echo(f"\nResults saved to {output_dir}/")


# ======================================================================
# Evaluate command
# ======================================================================


@main.command()
@click.option("--results", "results_dir", type=click.Path(exists=True), required=True,
              help="Path to results directory.")
@click.option("--true-dag", type=click.Path(exists=True), default=None,
              help="Path to ground-truth DAG (CSV adjacency matrix or DOT file).")
@click.option("--format", "output_format", type=click.Choice(["text", "json", "csv"]),
              default="text", help="Output format.")
def evaluate(results_dir: str, true_dag: Optional[str], output_format: str) -> None:
    """Evaluate experiment results against a ground-truth DAG.

    Computes SHD, F1, precision, recall, and QD metrics.
    """
    results_path = Path(results_dir) / "results.json"
    if not results_path.exists():
        raise click.ClickException(f"No results.json found in {results_dir}")

    results = json.loads(results_path.read_text())
    eval_results: dict = {}

    # QD metrics
    eval_results["qd_score"] = results.get("qd_score", "N/A")
    eval_results["coverage"] = results.get("coverage", "N/A")
    eval_results["n_elites"] = results.get("n_elites", "N/A")

    if true_dag is not None:
        true_adj = _load_ground_truth(true_dag)
        best_adj_raw = results.get("best_adjacency", [])
        if best_adj_raw:
            best_adj = np.array(best_adj_raw, dtype=np.int8)

            from causal_qd.metrics.structural import SHD, F1

            shd_val = SHD.compute(best_adj, true_adj)
            f1_metric = F1()
            f1_val = f1_metric.compute(best_adj, true_adj)
            prec_val = f1_metric.precision()
            recall_val = f1_metric.recall()

            eval_results["shd"] = shd_val
            eval_results["f1"] = f1_val
            eval_results["precision"] = prec_val
            eval_results["recall"] = recall_val

    if output_format == "json":
        click.echo(json.dumps(eval_results, indent=2, default=str))
    elif output_format == "csv":
        click.echo(_format_output(eval_results, "csv"))
    else:
        click.echo("=" * 40)
        click.echo("Evaluation Results")
        click.echo("=" * 40)
        for key, val in eval_results.items():
            if isinstance(val, float):
                click.echo(f"  {key:15s}: {val:.4f}")
            else:
                click.echo(f"  {key:15s}: {val}")
        click.echo("=" * 40)


# ======================================================================
# Visualize command
# ======================================================================


@main.command()
@click.option("--results", "results_dir", type=click.Path(exists=True), required=True,
              help="Path to results directory.")
@click.option("--output", "output_dir", type=click.Path(), default=None,
              help="Output directory for plots.")
@click.option("--format", "img_format", type=click.Choice(["png", "pdf", "svg"]),
              default="png", help="Image format.")
@click.option("--dpi", type=int, default=150, help="Plot resolution.")
def visualize(results_dir: str, output_dir: Optional[str], img_format: str, dpi: int) -> None:
    """Generate visualisation plots from experiment results.

    Creates convergence, coverage, QD-score, and summary plots.
    """
    if output_dir is None:
        output_dir = str(Path(results_dir) / "plots")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results_path = Path(results_dir) / "results.json"
    if not results_path.exists():
        raise click.ClickException(f"No results.json found in {results_dir}")

    results = json.loads(results_path.read_text())

    from causal_qd.visualization.convergence import ConvergencePlotter
    from causal_qd.visualization.archive_plotter import ArchivePlotter

    saved = []

    if "qd_score_history" in results:
        fig = ArchivePlotter.plot_qd_score_over_time(results["qd_score_history"])
        path = str(Path(output_dir) / f"qd_score.{img_format}")
        ArchivePlotter.save(fig, path, dpi=dpi)
        saved.append(path)

    if "coverage_history" in results:
        fig = ArchivePlotter.plot_coverage_over_time(results["coverage_history"])
        path = str(Path(output_dir) / f"coverage.{img_format}")
        ArchivePlotter.save(fig, path, dpi=dpi)
        saved.append(path)

    if "best_quality_history" in results:
        fig = ArchivePlotter.plot_best_quality_over_time(results["best_quality_history"])
        path = str(Path(output_dir) / f"best_quality.{img_format}")
        ArchivePlotter.save(fig, path, dpi=dpi)
        saved.append(path)

    # Convergence summary with all history metrics
    metrics_hist = {
        k.replace("_history", ""): v
        for k, v in results.items()
        if k.endswith("_history") and isinstance(v, list)
    }
    if metrics_hist:
        fig = ConvergencePlotter.plot_convergence_curves(metrics_hist)
        path = str(Path(output_dir) / f"convergence.{img_format}")
        ArchivePlotter.save(fig, path, dpi=dpi)
        saved.append(path)

    # Multi-panel summary
    if "coverage_history" in results and "qd_score_history" in results:
        bqh = results.get("best_quality_history", results["qd_score_history"])
        fig = ArchivePlotter.plot_summary(
            results["coverage_history"],
            results["qd_score_history"],
            bqh,
        )
        path = str(Path(output_dir) / f"summary.{img_format}")
        ArchivePlotter.save(fig, path, dpi=dpi)
        saved.append(path)

    for s in saved:
        click.echo(f"Saved: {s}")
    click.echo(f"Generated {len(saved)} plots in {output_dir}/")


# ======================================================================
# Benchmark command
# ======================================================================


@main.command()
@click.option("--n-nodes", type=int, default=5, help="Number of nodes.")
@click.option("--n-samples", type=int, default=1000, help="Number of data samples.")
@click.option("--n-generations", type=int, default=200, help="Number of generations.")
@click.option("--batch-size", type=int, default=20, help="Batch size.")
@click.option("--seed", type=int, default=42, help="Random seed.")
@click.option("--output", "output_dir", type=click.Path(), default="benchmark_results",
              help="Output directory.")
@click.option("--include-baselines/--no-baselines", default=True,
              help="Run baseline algorithms for comparison.")
def benchmark(
    n_nodes: int,
    n_samples: int,
    n_generations: int,
    batch_size: int,
    seed: int,
    output_dir: str,
    include_baselines: bool,
) -> None:
    """Run standard benchmark experiments.

    Generates synthetic data, runs CausalQD, and optionally runs
    baseline algorithms (PC, GES, Random) for comparison.
    """
    click.echo(f"Benchmark: {n_nodes} nodes, {n_samples} samples, seed={seed}")

    rng = np.random.default_rng(seed)

    # Generate synthetic linear SEM data
    click.echo("Generating synthetic data...")
    from causal_qd.baselines.random_dag import RandomDAGBaseline
    true_dag = RandomDAGBaseline.random_erdos_renyi(n_nodes, 0.3, rng)
    true_adj = true_dag.adjacency

    # Generate data from the DAG using linear SEM
    order = true_dag.topological_order
    data = np.zeros((n_samples, n_nodes))
    for node in order:
        parents = sorted(true_dag.parents(node))
        noise = rng.standard_normal(n_samples)
        if parents:
            weights = rng.uniform(0.5, 2.0, size=len(parents))
            data[:, node] = data[:, parents] @ weights + noise
        else:
            data[:, node] = noise

    click.echo(f"  True DAG has {true_dag.num_edges} edges")

    # Save true DAG
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    np.savetxt(Path(output_dir) / "true_dag.csv", true_adj, delimiter=",", fmt="%d")

    # Save data
    data_path = Path(output_dir) / "data.csv"
    header = ",".join(f"X{i}" for i in range(n_nodes))
    np.savetxt(data_path, data, delimiter=",", header=header, comments="")

    # Run CausalQD
    from causal_qd.config.config import ExperimentConfig
    cfg = CausalQDConfig(
        n_nodes=n_nodes,
        experiment=ExperimentConfig(
            n_iterations=n_generations,
            batch_size=batch_size,
            seed=seed,
            output_dir=output_dir,
        ),
    )

    from causal_qd.cli.experiment_runner import ExperimentRunner
    from causal_qd.cli.result_reporter import ResultReporter

    click.echo("Running CausalQD...")
    runner = ExperimentRunner(cfg)
    results = runner.run(data)

    # Evaluate against true DAG
    from causal_qd.metrics.structural import SHD, F1
    best_adj = np.array(results.get("best_adjacency", []))
    if best_adj.size:
        best_adj = best_adj.astype(np.int8)
        results["shd"] = int(SHD.compute(best_adj, true_adj))
        f1_m = F1()
        results["f1"] = float(f1_m.compute(best_adj, true_adj))
        results["precision"] = float(f1_m.precision())
        results["recall"] = float(f1_m.recall())

    # Baselines
    if include_baselines:
        click.echo("Running baselines...")
        baseline_results = {}

        # Random baseline
        rand_bl = RandomDAGBaseline(n_random=100, edge_prob=0.3)
        rand_dag = rand_bl.fit(data, rng=np.random.default_rng(seed))
        rand_adj = rand_dag.adjacency
        baseline_results["random"] = {
            "shd": int(SHD.compute(rand_adj, true_adj)),
            "f1": float(F1().compute(rand_adj, true_adj)),
        }
        click.echo(f"  Random: SHD={baseline_results['random']['shd']}")

        results["baselines"] = baseline_results

    reporter = ResultReporter(output_dir)
    reporter.report(results)
    click.echo(reporter.summary(results))


# ======================================================================
# Certificate command
# ======================================================================


@main.command()
@click.option("--results", "results_dir", type=click.Path(exists=True), required=True,
              help="Path to results directory.")
@click.option("--data", "data_path", type=click.Path(exists=True), required=True,
              help="Path to observational data CSV.")
@click.option("--n-bootstrap", type=int, default=100,
              help="Number of bootstrap samples.")
@click.option("--n-workers", type=int, default=1, help="Parallel workers.")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Output file for certificates.")
def certificate(
    results_dir: str,
    data_path: str,
    n_bootstrap: int,
    n_workers: int,
    output_path: Optional[str],
) -> None:
    """Compute robustness certificates for the best DAG in results.

    Uses bootstrap resampling to estimate the stability of each edge.
    """
    results_path = Path(results_dir) / "results.json"
    if not results_path.exists():
        raise click.ClickException(f"No results.json found in {results_dir}")

    results = json.loads(results_path.read_text())
    best_adj_raw = results.get("best_adjacency")
    if not best_adj_raw:
        raise click.ClickException("No best_adjacency in results")

    best_adj = np.array(best_adj_raw, dtype=np.int8)
    data = _load_data(data_path)
    n_nodes = best_adj.shape[0]

    click.echo(f"Computing certificates ({n_bootstrap} bootstrap samples)...")

    # Bootstrap edge stability
    rng = np.random.default_rng(42)
    n_samples = data.shape[0]
    edge_counts = np.zeros((n_nodes, n_nodes), dtype=np.float64)

    with click.progressbar(range(n_bootstrap), label="Bootstrap") as bar:
        for _ in bar:
            indices = rng.integers(0, n_samples, size=n_samples)
            boot_data = data[indices]

            # Simple OLS-based edge detection for bootstrap
            for j in range(n_nodes):
                parents = sorted(int(i) for i in np.nonzero(best_adj[:, j])[0])
                if not parents:
                    continue
                y = boot_data[:, j]
                X = boot_data[:, parents]
                try:
                    coef, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
                    # Keep edges with significant coefficients
                    rss_full = np.sum((y - X @ coef) ** 2)
                    for k, p in enumerate(parents):
                        X_red = np.delete(X, k, axis=1)
                        if X_red.shape[1] > 0:
                            coef_red, _, _, _ = np.linalg.lstsq(X_red, y, rcond=None)
                            rss_red = np.sum((y - X_red @ coef_red) ** 2)
                        else:
                            rss_red = np.sum((y - y.mean()) ** 2)
                        if rss_red > rss_full * 1.01:
                            edge_counts[p, j] += 1
                except np.linalg.LinAlgError:
                    pass

    certificates = edge_counts / max(n_bootstrap, 1)

    # Build certificate dict
    cert_dict = {}
    edges = list(zip(*np.nonzero(best_adj)))
    for i, j in edges:
        cert_dict[f"{i}->{j}"] = float(certificates[i, j])

    # Output
    if output_path is None:
        output_path = str(Path(results_dir) / "certificates.json")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(cert_dict, indent=2))

    click.echo(f"\nEdge certificates saved to {output_path}")
    click.echo("Edge certificates:")
    for edge_str, strength in sorted(cert_dict.items(), key=lambda x: -x[1]):
        marker = "✓" if strength >= 0.5 else "✗"
        click.echo(f"  {marker} {edge_str}: {strength:.3f}")

    strong = sum(1 for v in cert_dict.values() if v >= 0.5)
    click.echo(f"\n{strong}/{len(cert_dict)} edges above 0.5 threshold")


# ======================================================================
# From-pandas helper command
# ======================================================================


@main.command("from-pandas")
@click.option("--output", "output_path", type=click.Path(), default=None,
              help="Path for the exported file (default: stdout example).")
@click.option("--format", "export_format",
              type=click.Choice(["csv", "parquet", "npz"]),
              default="csv", help="Export format.")
def from_pandas(output_path: Optional[str], export_format: str) -> None:
    """Show how to export a pandas DataFrame for use with causal-qd.

    Prints a code snippet demonstrating DataFrame export to CSV, Parquet,
    or NPZ format suitable for ``causal-qd run --data``.
    """
    snippets = {
        "csv": (
            "import pandas as pd\n"
            "\n"
            "df = pd.read_csv('my_data.csv')  # or build your DataFrame\n"
            "# Select numeric columns and drop missing values\n"
            "numeric = df.select_dtypes(include='number').dropna()\n"
            "numeric.to_csv('causal_qd_input.csv', index=False)\n"
            "\n"
            "# Then run:\n"
            "#   causal-qd run --data causal_qd_input.csv --output results/"
        ),
        "parquet": (
            "import pandas as pd\n"
            "\n"
            "df = pd.read_csv('my_data.csv')  # or build your DataFrame\n"
            "numeric = df.select_dtypes(include='number').dropna()\n"
            "numeric.to_parquet('causal_qd_input.parquet', index=False)\n"
            "\n"
            "# Then run:\n"
            "#   causal-qd run --data causal_qd_input.parquet --output results/"
        ),
        "npz": (
            "import numpy as np\n"
            "import pandas as pd\n"
            "\n"
            "df = pd.read_csv('my_data.csv')  # or build your DataFrame\n"
            "numeric = df.select_dtypes(include='number').dropna()\n"
            "np.savez('causal_qd_input.npz', data=numeric.to_numpy())\n"
            "\n"
            "# Then run:\n"
            "#   causal-qd run --data causal_qd_input.npz --output results/"
        ),
    }
    click.echo(f"# Export pandas DataFrame to {export_format.upper()} for causal-qd\n")
    click.echo(snippets[export_format])

    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_text(snippets[export_format])
        click.echo(f"\nSnippet saved to {output_path}")


if __name__ == "__main__":
    main()
