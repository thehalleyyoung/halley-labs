"""Unit tests for usability_oracle.sarif.writer.

Tests SarifBuilder, SarifWriter, WriterOptions, and convenience functions
for SARIF output generation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Kind,
    Level,
    SarifLog,
)
from usability_oracle.sarif.writer import (
    SarifBuilder,
    SarifWriter,
    WriterOptions,
    sarif_to_string,
    write_sarif,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _simple_builder() -> SarifBuilder:
    """Return a builder with one rule and one result."""
    b = SarifBuilder("test-tool", "1.0.0")
    b.add_rule("R1", "rule-one", short_description="A rule")
    b.add_result(rule_id="R1", message="Found issue", level=Level.ERROR)
    return b


# ═══════════════════════════════════════════════════════════════════════════
# SarifBuilder — add_rule
# ═══════════════════════════════════════════════════════════════════════════


class TestBuilderAddRule:
    """Tests for SarifBuilder.add_rule."""

    def test_add_rule_returns_self(self) -> None:
        b = SarifBuilder("t", "1.0")
        ret = b.add_rule("R1", "rule")
        assert ret is b

    def test_duplicate_rule_ignored(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_rule("R1", "first")
        b.add_rule("R1", "second")
        log = b.build()
        rules = log.runs[0].tool.driver.rules
        assert len(rules) == 1
        assert rules[0].name == "first"

    def test_rule_registered(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_rule("R1", "name", short_description="desc")
        log = b.build()
        rule = log.runs[0].tool.driver.rules[0]
        assert rule.id == "R1"
        assert rule.short_description is not None
        assert rule.short_description.text == "desc"


# ═══════════════════════════════════════════════════════════════════════════
# SarifBuilder — add_result
# ═══════════════════════════════════════════════════════════════════════════


class TestBuilderAddResult:
    """Tests for SarifBuilder.add_result."""

    def test_add_result_returns_self(self) -> None:
        b = SarifBuilder("t", "1.0")
        ret = b.add_result(rule_id="R1", message="m")
        assert ret is b

    def test_auto_registers_rule(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_result(rule_id="R_NEW", message="m")
        log = b.build()
        rule_ids = [r.id for r in log.runs[0].tool.driver.rules]
        assert "R_NEW" in rule_ids

    def test_result_rule_index(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_rule("R1", "first")
        b.add_rule("R2", "second")
        b.add_result(rule_id="R2", message="m")
        log = b.build()
        assert log.runs[0].results[0].rule_index == 1

    def test_result_with_location(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_result(rule_id="R1", message="m", uri="file.py", start_line=42)
        log = b.build()
        result = log.runs[0].results[0]
        assert len(result.locations) == 1
        pl = result.locations[0].physical_location
        assert pl is not None
        assert pl.artifact_location is not None
        assert pl.artifact_location.uri == "file.py"
        assert pl.region is not None
        assert pl.region.start_line == 42

    def test_result_level_and_kind(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_result(rule_id="R1", message="m", level=Level.NOTE, kind=Kind.REVIEW)
        log = b.build()
        r = log.runs[0].results[0]
        assert r.level == Level.NOTE
        assert r.kind == Kind.REVIEW


# ═══════════════════════════════════════════════════════════════════════════
# SarifBuilder — add_artifact and set_invocation
# ═══════════════════════════════════════════════════════════════════════════


class TestBuilderArtifactAndInvocation:
    """Tests for add_artifact and set_invocation."""

    def test_add_artifact(self) -> None:
        b = SarifBuilder("t", "1.0")
        ret = b.add_artifact("src/app.js", mime_type="application/javascript")
        assert ret is b
        log = b.build()
        assert len(log.runs[0].artifacts) == 1
        assert log.runs[0].artifacts[0].mime_type == "application/javascript"

    def test_duplicate_artifact_ignored(self) -> None:
        b = SarifBuilder("t", "1.0")
        b.add_artifact("a.py")
        b.add_artifact("a.py")
        log = b.build()
        assert len(log.runs[0].artifacts) == 1

    def test_set_invocation(self) -> None:
        b = SarifBuilder("t", "1.0")
        ret = b.set_invocation(command_line="test --all", execution_successful=True)
        assert ret is b
        log = b.build()
        assert len(log.runs[0].invocations) == 1
        assert log.runs[0].invocations[0].command_line == "test --all"


# ═══════════════════════════════════════════════════════════════════════════
# SarifBuilder — build
# ═══════════════════════════════════════════════════════════════════════════


class TestBuilderBuild:
    """Tests for SarifBuilder.build."""

    def test_build_returns_sarif_log(self) -> None:
        log = _simple_builder().build()
        assert isinstance(log, SarifLog)

    def test_built_log_version(self) -> None:
        log = _simple_builder().build()
        assert log.version == SARIF_VERSION

    def test_built_log_schema(self) -> None:
        log = _simple_builder().build()
        assert log.schema_uri == SARIF_SCHEMA_URI

    def test_built_log_has_one_run(self) -> None:
        log = _simple_builder().build()
        assert len(log.runs) == 1

    def test_chain_methods(self) -> None:
        log = (
            SarifBuilder("t", "1.0")
            .add_rule("R1", "r")
            .add_result(rule_id="R1", message="m")
            .add_artifact("f.py")
            .set_invocation(command_line="cmd")
            .build()
        )
        assert isinstance(log, SarifLog)
        assert log.total_results == 1


# ═══════════════════════════════════════════════════════════════════════════
# SarifWriter.to_string
# ═══════════════════════════════════════════════════════════════════════════


class TestWriterToString:
    """Tests for SarifWriter.to_string producing valid JSON."""

    def test_produces_valid_json(self) -> None:
        log = _simple_builder().build()
        writer = SarifWriter()
        text = writer.to_string(log)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)
        assert parsed["version"] == SARIF_VERSION

    def test_pretty_mode_has_indent(self) -> None:
        log = _simple_builder().build()
        writer = SarifWriter(WriterOptions(pretty=True, indent=2))
        text = writer.to_string(log)
        assert "\n" in text
        assert "  " in text

    def test_compact_mode_no_indent(self) -> None:
        log = _simple_builder().build()
        writer = SarifWriter(WriterOptions(pretty=False))
        text = writer.to_string(log)
        lines = text.strip().split("\n")
        assert len(lines) == 1


# ═══════════════════════════════════════════════════════════════════════════
# WriterOptions
# ═══════════════════════════════════════════════════════════════════════════


class TestWriterOptions:
    """Tests for WriterOptions: sort_keys, omit_empty."""

    def test_default_options(self) -> None:
        opts = WriterOptions()
        assert opts.pretty is True
        assert opts.sort_keys is False
        assert opts.omit_empty is True

    def test_sort_keys(self) -> None:
        log = _simple_builder().build()
        writer = SarifWriter(WriterOptions(sort_keys=True, pretty=True))
        text = writer.to_string(log)
        parsed = json.loads(text)
        assert isinstance(parsed, dict)

    def test_omit_empty(self) -> None:
        log = SarifBuilder("t", "1.0").build()
        writer = SarifWriter(WriterOptions(omit_empty=True))
        text = writer.to_string(log)
        parsed = json.loads(text)
        run = parsed["runs"][0]
        # Empty optional fields should not be present
        assert "artifacts" not in run or run["artifacts"] == []


# ═══════════════════════════════════════════════════════════════════════════
# write_sarif to file
# ═══════════════════════════════════════════════════════════════════════════


class TestWriteSarifToFile:
    """Tests for write_sarif convenience function."""

    def test_write_sarif_creates_file(self, tmp_path: Path) -> None:
        log = _simple_builder().build()
        path = tmp_path / "out.sarif"
        write_sarif(log, path)
        assert path.exists()
        parsed = json.loads(path.read_text(encoding="utf-8"))
        assert parsed["version"] == SARIF_VERSION

    def test_write_sarif_pretty(self, tmp_path: Path) -> None:
        log = _simple_builder().build()
        path = tmp_path / "pretty.sarif"
        write_sarif(log, path, pretty=True)
        text = path.read_text(encoding="utf-8")
        assert "\n" in text


# ═══════════════════════════════════════════════════════════════════════════
# sarif_to_string convenience
# ═══════════════════════════════════════════════════════════════════════════


class TestSarifToString:
    """Tests for sarif_to_string convenience function."""

    def test_returns_valid_json(self) -> None:
        log = _simple_builder().build()
        text = sarif_to_string(log)
        parsed = json.loads(text)
        assert parsed["version"] == SARIF_VERSION

    def test_pretty_param(self) -> None:
        log = _simple_builder().build()
        compact = sarif_to_string(log, pretty=False)
        pretty = sarif_to_string(log, pretty=True)
        assert len(pretty) > len(compact)
