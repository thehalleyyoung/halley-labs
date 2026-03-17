"""CLI module: command-line interface for RegSynth.

Provides subcommands: analyze, check, encode, solve, pareto, plan,
certify, verify, benchmark, export, visualize, report.
"""

from regsynth_py.cli.main import main
from regsynth_py.cli.config import load_config, Config

__all__ = ["main", "load_config", "Config"]
