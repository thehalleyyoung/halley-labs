"""
usability_oracle.cli.main — CLI entry point using Click.

Provides the ``usability-oracle`` command group with subcommands for
diff, analyze, benchmark, validate, and init operations.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import click

from usability_oracle.core.enums import OutputFormat


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ---------------------------------------------------------------------------
# Common options
# ---------------------------------------------------------------------------

_config_option = click.option(
    "--config", "-c",
    type=click.Path(exists=False),
    default=None,
    help="Path to YAML configuration file.",
)

_output_format_option = click.option(
    "--output-format", "-f",
    type=click.Choice(["json", "sarif", "html", "console"], case_sensitive=False),
    default="console",
    help="Output format.",
)

_verbose_option = click.option(
    "--verbose", "-v",
    is_flag=True,
    default=False,
    help="Enable verbose logging.",
)

_beta_range_option = click.option(
    "--beta-range",
    type=(float, float),
    default=(0.1, 20.0),
    help="Rationality parameter β range (min, max).",
)

_output_option = click.option(
    "--output", "-o",
    type=click.Path(),
    default=None,
    help="Output file path (stdout if not specified).",
)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(version="0.1.0", prog_name="usability-oracle")
def cli() -> None:
    """Bounded-Rational Usability Regression Oracle.

    Detect, quantify, and repair usability regressions using
    information-theoretic cognitive cost analysis.
    """
    pass


# ---------------------------------------------------------------------------
# diff command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("before", type=click.Path(exists=True))
@click.argument("after", type=click.Path(exists=True))
@click.option(
    "--task-spec", "-t",
    type=click.Path(exists=True),
    default=None,
    help="Task specification file (YAML/JSON).",
)
@_config_option
@_output_format_option
@_verbose_option
@_beta_range_option
@_output_option
def diff(
    before: str,
    after: str,
    task_spec: Optional[str],
    config: Optional[str],
    output_format: str,
    verbose: bool,
    beta_range: tuple[float, float],
    output: Optional[str],
) -> None:
    """Compare two UI versions for usability regressions.

    BEFORE and AFTER are paths to HTML or JSON accessibility-tree files.
    """
    _setup_logging(verbose)

    from usability_oracle.cli.commands.diff import diff_command

    exit_code = diff_command(
        before=before,
        after=after,
        task_spec=task_spec,
        config=config,
        output_format=output_format,
        beta_range=beta_range,
        output_path=output,
    )
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# analyze command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("source", type=click.Path(exists=True))
@click.option(
    "--task-spec", "-t",
    type=click.Path(exists=True),
    default=None,
    help="Task specification file.",
)
@_config_option
@_output_format_option
@_verbose_option
@_output_option
def analyze(
    source: str,
    task_spec: Optional[str],
    config: Optional[str],
    output_format: str,
    verbose: bool,
    output: Optional[str],
) -> None:
    """Analyse a single UI for usability issues.

    SOURCE is a path to an HTML or JSON accessibility-tree file.
    """
    _setup_logging(verbose)

    from usability_oracle.cli.commands.analyze import analyze_command

    exit_code = analyze_command(
        source=source,
        task_spec=task_spec,
        config=config,
        output_format=output_format,
        output_path=output,
    )
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# benchmark command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--suite", "-s",
    type=click.Choice(["small", "medium", "large", "all"]),
    default="small",
    help="Benchmark suite to run.",
)
@click.option(
    "--n-runs", "-n",
    type=int,
    default=3,
    help="Number of runs per benchmark.",
)
@_output_option
@_verbose_option
def benchmark(
    suite: str,
    n_runs: int,
    output: Optional[str],
    verbose: bool,
) -> None:
    """Run the benchmark suite."""
    _setup_logging(verbose)

    from usability_oracle.cli.commands.benchmark import benchmark_command

    exit_code = benchmark_command(
        suite=suite,
        n_runs=n_runs,
        output=output,
    )
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# validate command
# ---------------------------------------------------------------------------

@cli.command()
@click.argument("task_spec_file", type=click.Path(exists=True))
@click.option(
    "--ui-source",
    type=click.Path(exists=True),
    default=None,
    help="Optional UI source to validate against.",
)
@_verbose_option
def validate(
    task_spec_file: str,
    ui_source: Optional[str],
    verbose: bool,
) -> None:
    """Validate a task specification file.

    Checks syntax, references, and optionally verifies against a UI source.
    """
    _setup_logging(verbose)

    from usability_oracle.cli.commands.validate import validate_command

    exit_code = validate_command(
        task_spec_file=task_spec_file,
        ui_source=ui_source,
    )
    sys.exit(exit_code)


# ---------------------------------------------------------------------------
# init command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--output-dir", "-d",
    type=click.Path(),
    default=".",
    help="Directory to write config files.",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing config files.",
)
def init(output_dir: str, force: bool) -> None:
    """Initialise a usability-oracle configuration.

    Creates a default ``usability-oracle.yaml`` in the target directory.
    """
    from usability_oracle.pipeline.config import FullPipelineConfig

    out_path = Path(output_dir) / "usability-oracle.yaml"
    if out_path.exists() and not force:
        click.echo(f"Config already exists: {out_path}  (use --force to overwrite)")
        sys.exit(1)

    import yaml

    config = FullPipelineConfig.DEFAULT()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.dump(config.to_dict(), f, default_flow_style=False, sort_keys=False)

    click.echo(f"Created config: {out_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Main entry point for the CLI."""
    cli()


if __name__ == "__main__":
    main()
