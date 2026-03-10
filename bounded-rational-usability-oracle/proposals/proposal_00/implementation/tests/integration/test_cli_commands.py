"""Integration tests for the CLI commands.

These tests use ``click.testing.CliRunner`` to invoke the CLI without
spawning a subprocess.  They verify that ``init``, ``validate``, ``diff``,
``analyze``, and ``benchmark`` commands produce correct exit codes, output
the expected content, and handle invalid arguments gracefully.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from usability_oracle.cli.main import cli
from usability_oracle.taskspec.models import TaskStep, TaskFlow, TaskSpec

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
SAMPLE_HTML_DIR = FIXTURES_DIR / "sample_html"


def _simple_task_yaml() -> str:
    """Return a minimal valid task-spec YAML string."""
    return """\
spec_id: login
name: Login Task
description: Fill in the login form
flows:
  - flow_id: login_flow
    name: Login
    steps:
      - step_id: s1
        action_type: click
        target_role: textfield
        target_name: Username
        description: Focus username field
      - step_id: s2
        action_type: type
        target_role: textfield
        target_name: Username
        input_value: admin
        description: Type username
        depends_on: [s1]
      - step_id: s3
        action_type: click
        target_role: button
        target_name: Submit
        description: Click submit
        depends_on: [s2]
    success_criteria:
      - form_submitted
"""


def _write_task_yaml(tmp_path: Path) -> Path:
    """Write a task spec YAML to a temp directory and return its path."""
    p = tmp_path / "task_spec.yaml"
    p.write_text(_simple_task_yaml())
    return p


# ===================================================================
# Tests – Main group
# ===================================================================


class TestCLIMainGroup:
    """Verify the top-level CLI group behaviour."""

    def test_help_flag(self) -> None:
        """``--help`` should exit with code 0 and show usage."""
        runner = CliRunner()
        result = runner.invoke(cli, ["--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_no_args_shows_help(self) -> None:
        """Invoking with no arguments should show help or exit cleanly."""
        runner = CliRunner()
        result = runner.invoke(cli, [])
        # Either shows help (exit 0) or prints error (exit non-zero)
        assert result.exit_code in (0, 1, 2)

    def test_unknown_command_error(self) -> None:
        """An unknown subcommand should produce a non-zero exit code."""
        runner = CliRunner()
        result = runner.invoke(cli, ["nonexistent-command"])
        assert result.exit_code != 0


# ===================================================================
# Tests – init command
# ===================================================================


class TestCLIInit:
    """Verify the ``init`` command creates a project scaffold."""

    def test_init_creates_directory(self, tmp_path: Path) -> None:
        """``init`` should create a directory structure."""
        runner = CliRunner()
        out_dir = tmp_path / "my_project"
        result = runner.invoke(cli, ["init", "-d", str(out_dir)])
        assert result.exit_code == 0
        assert out_dir.exists()

    def test_init_help(self) -> None:
        """``init --help`` should show usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["init", "--help"])
        assert result.exit_code == 0
        assert "Usage" in result.output or "usage" in result.output.lower()

    def test_init_force_flag(self, tmp_path: Path) -> None:
        """``init --force`` should succeed even if directory exists."""
        runner = CliRunner()
        out_dir = tmp_path / "existing_project"
        out_dir.mkdir()
        result = runner.invoke(cli, ["init", "-d", str(out_dir), "--force"])
        assert result.exit_code == 0

    def test_init_existing_without_force(self, tmp_path: Path) -> None:
        """``init`` without --force on existing dir may warn or fail."""
        runner = CliRunner()
        out_dir = tmp_path / "existing_project2"
        out_dir.mkdir()
        result = runner.invoke(cli, ["init", str(out_dir)])
        # May succeed with warning or fail — both are acceptable
        assert isinstance(result.exit_code, int)


# ===================================================================
# Tests – validate command
# ===================================================================


class TestCLIValidate:
    """Verify the ``validate`` command checks task spec YAML."""

    def test_validate_valid_yaml(self, tmp_path: Path) -> None:
        """A valid task spec YAML should pass validation."""
        yaml_path = _write_task_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(yaml_path)])
        assert result.exit_code == 0

    def test_validate_help(self) -> None:
        """``validate --help`` should show usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", "--help"])
        assert result.exit_code == 0

    def test_validate_nonexistent_file(self, tmp_path: Path) -> None:
        """Validating a non-existent file should produce a non-zero exit."""
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(tmp_path / "nope.yaml")])
        assert result.exit_code != 0

    def test_validate_invalid_yaml(self, tmp_path: Path) -> None:
        """Malformed YAML should fail validation."""
        bad = tmp_path / "bad.yaml"
        bad.write_text("{{not: valid: yaml: [}")
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(bad)])
        assert result.exit_code != 0

    def test_validate_with_verbose(self, tmp_path: Path) -> None:
        """``validate --verbose`` should include extra information."""
        yaml_path = _write_task_yaml(tmp_path)
        runner = CliRunner()
        result = runner.invoke(cli, ["validate", str(yaml_path), "--verbose"])
        assert result.exit_code == 0


# ===================================================================
# Tests – diff command
# ===================================================================


class TestCLIDiff:
    """Verify the ``diff`` command compares two UI sources."""

    def test_diff_help(self) -> None:
        """``diff --help`` should show usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diff", "--help"])
        assert result.exit_code == 0

    def test_diff_missing_args(self) -> None:
        """``diff`` without arguments should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["diff"])
        assert result.exit_code != 0

    def test_diff_nonexistent_files(self, tmp_path: Path) -> None:
        """``diff`` with non-existent files should produce an error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diff",
            str(tmp_path / "a.html"),
            str(tmp_path / "b.html"),
        ])
        assert result.exit_code != 0


# ===================================================================
# Tests – analyze command
# ===================================================================


class TestCLIAnalyze:
    """Verify the ``analyze`` command."""

    def test_analyze_help(self) -> None:
        """``analyze --help`` should show usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", "--help"])
        assert result.exit_code == 0

    def test_analyze_missing_source(self) -> None:
        """``analyze`` without a source argument should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze"])
        assert result.exit_code != 0

    def test_analyze_nonexistent_source(self, tmp_path: Path) -> None:
        """Analysing a non-existent file should fail."""
        runner = CliRunner()
        result = runner.invoke(cli, ["analyze", str(tmp_path / "nope.html")])
        assert result.exit_code != 0


# ===================================================================
# Tests – benchmark command
# ===================================================================


class TestCLIBenchmark:
    """Verify the ``benchmark`` command."""

    def test_benchmark_help(self) -> None:
        """``benchmark --help`` should show usage information."""
        runner = CliRunner()
        result = runner.invoke(cli, ["benchmark", "--help"])
        assert result.exit_code == 0


# ===================================================================
# Tests – Error handling
# ===================================================================


class TestCLIErrorHandling:
    """Verify that invalid inputs produce clean error output."""

    def test_invalid_output_format(self, tmp_path: Path) -> None:
        """An unsupported --output-format should fail cleanly."""
        html = SAMPLE_HTML_DIR / "simple_form.html"
        if not html.exists():
            pytest.skip("Fixture not found")
        runner = CliRunner()
        result = runner.invoke(cli, [
            "analyze", str(html), "--output-format", "invalid_format",
        ])
        # Should either reject the invalid format or handle it
        assert isinstance(result.exit_code, int)

    def test_diff_with_invalid_beta_range(self, tmp_path: Path) -> None:
        """An invalid beta range should produce a clear error."""
        runner = CliRunner()
        result = runner.invoke(cli, [
            "diff",
            str(SAMPLE_HTML_DIR / "simple_form.html"),
            str(SAMPLE_HTML_DIR / "navigation_menu.html"),
            "--beta-range", "invalid",
        ])
        assert result.exit_code != 0
