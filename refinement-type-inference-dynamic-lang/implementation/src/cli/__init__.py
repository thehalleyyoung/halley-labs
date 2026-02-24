from __future__ import annotations

"""
reftype.cli — CLI package for the refinement type inference system.

Provides command-line tools for analyzing Python and TypeScript code,
running in CI pipelines, and serving as an LSP server for editor
integration.  Built around CEGAR-based refinement type inference for
dynamically-typed languages.
"""

__version__ = "0.1.0"

from dataclasses import dataclass
from typing import Optional, Sequence


@dataclass(frozen=True)
class CliInfo:
    """Static metadata exposed by the package."""

    name: str = "reftype"
    version: str = __version__
    description: str = (
        "Refinement type inference for dynamically-typed languages "
        "(Python & TypeScript) using CEGAR"
    )
    author: str = "reftype contributors"
    license: str = "MIT"
    homepage: str = "https://github.com/reftype/reftype"
    supported_languages: tuple[str, ...] = ("python", "typescript")
    default_config_files: tuple[str, ...] = (
        ".reftype.toml",
        "pyproject.toml",
        "package.json",
    )


CLI_INFO = CliInfo()

_ENTRY_POINTS: dict[str, str] = {
    "analyze": "src.cli.main:AnalyzeCommand",
    "watch": "src.cli.main:WatchCommand",
    "ci-check": "src.cli.main:CiCheckCommand",
    "init": "src.cli.main:InitCommand",
    "report": "src.cli.main:ReportCommand",
    "export": "src.cli.main:ExportCommand",
    "diff": "src.cli.main:DiffCommand",
    "server": "src.cli.main:ServerCommand",
    "version": "src.cli.main:VersionCommand",
    "config": "src.cli.main:ConfigCommand",
}


def entry_point(argv: Optional[Sequence[str]] = None) -> int:
    """Main entry point invoked by the ``reftype`` console script."""
    from src.cli.main import ReftypeCliApp  # local to avoid circular imports

    app = ReftypeCliApp()
    return app.run(argv)
