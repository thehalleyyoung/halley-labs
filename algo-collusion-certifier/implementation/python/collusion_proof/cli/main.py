"""CollusionProof CLI entry point."""

import click
import json
import sys
import numpy as np
from pathlib import Path

from collusion_proof.cli.commands import (
    load_price_data,
    save_results,
    format_text_output,
    format_json_output,
    format_verdict_display,
    print_banner,
    validate_inputs,
    run_analysis,
    run_benchmark,
    run_simulation,
    generate_certificate,
    run_parameter_sweep,
    ProgressDisplay,
)


@click.group()
@click.version_option(version="0.1.0")
@click.option("--verbose", "-v", is_flag=True, help="Enable verbose output")
@click.option("--config", "-c", type=click.Path(exists=True), help="Config file path")
@click.pass_context
def cli(ctx, verbose, config):
    """CollusionProof: Algorithmic Collusion Certification System.

    Detect and certify algorithmic collusion in automated pricing systems.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    if config:
        from collusion_proof.config import load_config

        cfg = load_config(config)
        ctx.obj["config"] = cfg
    else:
        from collusion_proof.config import get_config

        ctx.obj["config"] = get_config()

    if verbose:
        from collusion_proof.utils import setup_logging

        setup_logging("DEBUG")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--nash-price", type=float, required=True, help="Nash equilibrium price")
@click.option("--monopoly-price", type=float, required=True, help="Monopoly price")
@click.option("--alpha", type=float, default=0.05, help="Significance level")
@click.option("--bootstrap-samples", type=int, default=10000, help="Bootstrap samples")
@click.option("--output", "-o", type=click.Path(), help="Output report path")
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["text", "json", "html"]),
    default="text",
)
@click.pass_context
def analyze(
    ctx,
    data_path,
    nash_price,
    monopoly_price,
    alpha,
    bootstrap_samples,
    output,
    output_format,
):
    """Analyze price data for collusion."""
    verbose = ctx.obj.get("verbose", False)
    print_banner()

    try:
        prices = load_price_data(data_path)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    warnings = validate_inputs(nash_price, monopoly_price, prices)
    for w in warnings:
        click.echo(click.style(f"Warning: {w}", fg="yellow"), err=True)

    click.echo(
        f"Analyzing {prices.shape[0]} rounds, {prices.shape[1]} players..."
    )
    click.echo(
        f"Nash price: {nash_price:.4f}, Monopoly price: {monopoly_price:.4f}"
    )

    results = run_analysis(
        prices,
        nash_price,
        monopoly_price,
        alpha=alpha,
        bootstrap_samples=bootstrap_samples,
        verbose=verbose,
    )

    verdict = results.get("verdict", "inconclusive")
    confidence = results.get("confidence", 0.0)
    click.echo()
    click.echo(format_verdict_display(verdict, confidence))
    click.echo()

    if output_format == "text":
        report = format_text_output(results)
        click.echo(report)
    elif output_format == "json":
        report = format_json_output(results)
        click.echo(report)
    elif output_format == "html":
        report = _generate_html_report(results)
        if output:
            Path(output).write_text(report)
            click.echo(f"HTML report saved to {output}")
        else:
            click.echo(report)
        return

    if output:
        save_results(results, output, format=output_format)
        click.echo(f"Results saved to {output}")


@cli.command()
@click.option(
    "--mode",
    type=click.Choice(["smoke", "standard", "full"]),
    default="smoke",
)
@click.option("--seed", type=int, default=42)
@click.option("--output", "-o", type=click.Path())
@click.pass_context
def benchmark(ctx, mode, seed, output):
    """Run evaluation benchmarks."""
    verbose = ctx.obj.get("verbose", False)
    print_banner()
    click.echo(f"Running {mode} benchmark (seed={seed})...")

    results = run_benchmark(mode, seed=seed, verbose=verbose)

    accuracy = results.get("accuracy", 0.0)
    total = results.get("total_scenarios", 0)
    correct = results.get("correct", 0)
    type_i = results.get("type_i_error_rate", 0.0)
    type_ii = results.get("type_ii_error_rate", 0.0)

    click.echo()
    click.echo(f"  Scenarios : {total}")
    click.echo(f"  Correct   : {correct}/{total}")
    click.echo(f"  Accuracy  : {accuracy:.2%}")
    click.echo(f"  Type I    : {type_i:.2%}")
    click.echo(f"  Type II   : {type_ii:.2%}")

    if "precision" in results:
        click.echo(f"  Precision : {results['precision']:.2%}")
        click.echo(f"  Recall    : {results['recall']:.2%}")
        click.echo(f"  F1 Score  : {results['f1_score']:.2%}")

    if output:
        save_results(results, output, format="json")
        click.echo(f"\nResults saved to {output}")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--nash-price", type=float, required=True)
@click.option("--monopoly-price", type=float, required=True)
@click.option("--output-dir", "-o", type=click.Path(), default="./plots")
@click.pass_context
def visualize(ctx, data_path, nash_price, monopoly_price, output_dir):
    """Generate visualizations from price data."""
    print_banner()

    try:
        prices = load_price_data(data_path)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    num_rounds, num_players = prices.shape

    # Price trajectories
    fig, ax = plt.subplots(figsize=(12, 6))
    for p in range(num_players):
        ax.plot(prices[:, p], alpha=0.7, label=f"Player {p}")
    ax.axhline(nash_price, color="green", linestyle="--", label="Nash")
    ax.axhline(monopoly_price, color="red", linestyle="--", label="Monopoly")
    ax.set_xlabel("Round")
    ax.set_ylabel("Price")
    ax.set_title("Price Trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "price_trajectories.png", dpi=150)
    plt.close(fig)
    click.echo(f"  Saved price_trajectories.png")

    # Price distribution
    fig, ax = plt.subplots(figsize=(10, 6))
    for p in range(num_players):
        ax.hist(prices[:, p], bins=50, alpha=0.5, label=f"Player {p}")
    ax.axvline(nash_price, color="green", linestyle="--", label="Nash")
    ax.axvline(monopoly_price, color="red", linestyle="--", label="Monopoly")
    ax.set_xlabel("Price")
    ax.set_ylabel("Frequency")
    ax.set_title("Price Distribution")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "price_distribution.png", dpi=150)
    plt.close(fig)
    click.echo(f"  Saved price_distribution.png")

    # Convergence plot (rolling mean)
    fig, ax = plt.subplots(figsize=(12, 6))
    window = max(num_rounds // 100, 10)
    for p in range(num_players):
        rolling = np.convolve(prices[:, p], np.ones(window) / window, mode="valid")
        ax.plot(rolling, alpha=0.7, label=f"Player {p}")
    ax.axhline(nash_price, color="green", linestyle="--", label="Nash")
    ax.axhline(monopoly_price, color="red", linestyle="--", label="Monopoly")
    ax.set_xlabel("Round")
    ax.set_ylabel("Rolling Mean Price")
    ax.set_title(f"Price Convergence (window={window})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "convergence.png", dpi=150)
    plt.close(fig)
    click.echo(f"  Saved convergence.png")

    click.echo(f"\nAll plots saved to {output_dir}")


@cli.command()
@click.option("--num-players", type=int, default=2)
@click.option("--num-rounds", type=int, default=100000)
@click.option(
    "--algorithm",
    type=click.Choice(["q_learning", "dqn", "grim_trigger", "bandit"]),
    default="q_learning",
)
@click.option("--seed", type=int, default=None)
@click.option("--output", "-o", type=click.Path())
@click.pass_context
def simulate(ctx, num_players, num_rounds, algorithm, seed, output):
    """Run a pricing game simulation."""
    verbose = ctx.obj.get("verbose", False)
    print_banner()

    click.echo(
        f"Simulating {algorithm} with {num_players} players, "
        f"{num_rounds} rounds..."
    )

    results = run_simulation(num_players, num_rounds, algorithm, seed=seed)

    prices = results.get("prices")
    if prices is not None:
        mean_price = float(np.mean(prices))
        final_mean = float(np.mean(prices[-1000:]))
        click.echo(f"  Overall mean price : {mean_price:.4f}")
        click.echo(f"  Final 1000 mean    : {final_mean:.4f}")
        click.echo(f"  Nash price         : {results.get('nash_price', 'N/A')}")
        click.echo(f"  Monopoly price     : {results.get('monopoly_price', 'N/A')}")

    if output:
        out_path = Path(output)
        if out_path.suffix == ".npy":
            np.save(out_path, prices)
        elif out_path.suffix == ".csv":
            np.savetxt(out_path, prices, delimiter=",")
        else:
            save_results(
                {k: v for k, v in results.items() if k != "prices"},
                output,
                format="json",
            )
            if prices is not None:
                np.save(str(out_path.with_suffix(".npy")), prices)
        click.echo(f"  Results saved to {output}")


@cli.command()
@click.argument("data_path", type=click.Path(exists=True))
@click.option("--nash-price", type=float, required=True)
@click.option("--monopoly-price", type=float, required=True)
@click.option("--system-id", type=str, default="SYSTEM-001")
@click.option("--output", "-o", type=click.Path(), default="./certificate.html")
@click.pass_context
def certify(ctx, data_path, nash_price, monopoly_price, system_id, output):
    """Generate a formal certification report."""
    verbose = ctx.obj.get("verbose", False)
    print_banner()

    try:
        prices = load_price_data(data_path)
    except Exception as e:
        click.echo(f"Error loading data: {e}", err=True)
        sys.exit(1)

    click.echo(f"Generating certificate for system {system_id}...")

    results = run_analysis(
        prices,
        nash_price,
        monopoly_price,
        verbose=verbose,
    )

    cert_path = generate_certificate(results, system_id, output)
    click.echo(f"\nCertificate generated: {cert_path}")
    click.echo(format_verdict_display(results["verdict"], results["confidence"]))


@cli.command()
@click.option(
    "--param",
    "-p",
    multiple=True,
    help="param=start:stop:steps format",
)
@click.option("--metric", type=str, default="accuracy")
@click.option("--output", "-o", type=click.Path())
@click.pass_context
def sweep(ctx, param, metric, output):
    """Run parameter sweep analysis."""
    verbose = ctx.obj.get("verbose", False)
    print_banner()

    if not param:
        click.echo("No parameters specified. Use -p 'param=start:stop:steps'", err=True)
        sys.exit(1)

    click.echo(f"Running parameter sweep on metric: {metric}")
    for p in param:
        click.echo(f"  {p}")

    results = run_parameter_sweep(list(param), metric, verbose=verbose)

    best = results.get("best_params", {})
    best_score = results.get("best_score", 0.0)
    click.echo(f"\nBest {metric}: {best_score:.4f}")
    click.echo(f"Best params: {json.dumps(best, indent=2)}")

    if output:
        save_results(results, output, format="json")
        click.echo(f"Results saved to {output}")


@cli.command()
@click.pass_context
def info(ctx):
    """Show system information and configuration."""
    from collusion_proof import __version__
    from collusion_proof.config import get_config

    print_banner()
    config = ctx.obj.get("config", get_config())

    click.echo(f"Version     : {__version__}")
    click.echo(f"Python      : {sys.version.split()[0]}")
    click.echo(f"NumPy       : {np.__version__}")

    try:
        import scipy

        click.echo(f"SciPy       : {scipy.__version__}")
    except ImportError:
        click.echo("SciPy       : not installed")

    try:
        import matplotlib

        click.echo(f"Matplotlib  : {matplotlib.__version__}")
    except ImportError:
        click.echo("Matplotlib  : not installed")

    click.echo()
    click.echo("Configuration:")
    click.echo(f"  Alpha             : {config.test.alpha}")
    click.echo(f"  Bootstrap samples : {config.test.bootstrap_samples}")
    click.echo(f"  Num players       : {config.market.num_players}")
    click.echo(f"  Num rounds        : {config.market.num_rounds}")
    click.echo(f"  Marginal cost     : {config.market.marginal_cost}")
    click.echo(f"  Nash price        : {config.market.nash_price:.4f}")
    click.echo(f"  Monopoly price    : {config.market.monopoly_price:.4f}")
    click.echo(f"  Random seed       : {config.random_seed}")
    click.echo(f"  Verbose           : {config.verbose}")


def _generate_html_report(results: dict) -> str:
    """Generate a minimal HTML report from analysis results."""
    verdict = results.get("verdict", "inconclusive")
    confidence = results.get("confidence", 0.0)
    premium = results.get("collusion_premium", 0.0)
    collusion_index = results.get("collusion_index", 0.0)

    color_map = {
        "competitive": "#28a745",
        "suspicious": "#ffc107",
        "collusive": "#dc3545",
        "inconclusive": "#6c757d",
    }
    color = color_map.get(verdict, "#6c757d")

    tier_rows = ""
    for tier in results.get("tier_results", []):
        tier_name = tier.get("tier", "unknown")
        tier_reject = tier.get("combined_reject", False)
        tier_p = tier.get("combined_p_value", 1.0)
        status = "REJECT" if tier_reject else "ACCEPT"
        tier_rows += (
            f"<tr><td>{tier_name}</td>"
            f"<td>{status}</td>"
            f"<td>{tier_p:.4g}</td></tr>\n"
        )

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>CollusionProof Report</title>
<style>
  body {{ font-family: sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
  .verdict {{ font-size: 2em; color: {color}; font-weight: bold; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
  th {{ background-color: #f4f4f4; }}
</style></head><body>
<h1>CollusionProof Certification Report</h1>
<p class="verdict">Verdict: {verdict.upper()}</p>
<p>Confidence: {confidence:.2%}</p>
<p>Collusion Premium: {premium:.4f}</p>
<p>Collusion Index: {collusion_index:.4f}</p>
<h2>Tier Results</h2>
<table>
<tr><th>Tier</th><th>Decision</th><th>p-value</th></tr>
{tier_rows}
</table>
</body></html>"""
    return html


def main():
    cli(obj={})


if __name__ == "__main__":
    main()
