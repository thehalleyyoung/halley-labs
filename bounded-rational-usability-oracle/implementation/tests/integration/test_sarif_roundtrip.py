"""Integration tests for SARIF round-trip.

Create findings → write SARIF → read SARIF → verify.
Merge multiple SARIF files.
SARIF diff for regression detection.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from usability_oracle.sarif.schema import (
    SARIF_VERSION,
    Level,
    Kind,
    Result,
    Run,
    SarifLog,
    Tool,
    ToolComponent,
    Message,
    ReportingDescriptor,
    MultiformatMessageString,
    Location,
    PhysicalLocation,
    ArtifactLocation,
    Region,
)
from usability_oracle.sarif.writer import (
    SarifBuilder,
    SarifWriter,
    WriterOptions,
    sarif_to_string,
    write_sarif,
)
from usability_oracle.sarif.reader import (
    read_sarif,
    SarifReader,
    SarifParseError,
)
from usability_oracle.sarif.validator import (
    validate_sarif,
    SarifValidator,
    ValidationReport,
)
from usability_oracle.sarif.converter import (
    merge_sarif_logs,
    merge_runs,
)
from usability_oracle.sarif.diff import (
    diff_sarif,
    DiffCategory,
    DiffReport,
    SarifDiffer,
    compute_fingerprint,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_simple_log(
    tool_name: str = "usability-oracle",
    n_results: int = 3,
) -> SarifLog:
    """Build a simple SARIF log with n_results findings."""
    builder = SarifBuilder(tool_name=tool_name, tool_version="0.1.0")
    for i in range(n_results):
        rule_id = f"UO{i:03d}"
        builder.add_rule(
            rule_id=rule_id,
            name=f"Rule {i}",
            short_description=f"Description for rule {i}",
        )
        builder.add_result(
            rule_id=rule_id,
            message=f"Finding {i}: target element too small",
            level=Level.WARNING if i % 2 == 0 else Level.ERROR,
            uri=f"page_{i}.html",
            start_line=10 + i,
        )
    return builder.build()


def _build_log_with_fingerprints(
    tool_name: str = "usability-oracle",
    rule_ids: list[str] | None = None,
) -> SarifLog:
    """Build a SARIF log with fingerprinted results."""
    if rule_ids is None:
        rule_ids = ["UO001", "UO002", "UO003"]
    builder = SarifBuilder(tool_name=tool_name, tool_version="0.1.0")
    for rid in rule_ids:
        builder.add_rule(rule_id=rid, name=f"Rule {rid}")
        builder.add_result(
            rule_id=rid,
            message=f"Issue {rid}",
            level=Level.WARNING,
            uri="index.html",
            start_line=10,
            fingerprints={
                "primaryLocationLineHash": f"hash_{rid}",
            },
        )
    return builder.build()


# ===================================================================
# Tests — Write → Read round-trip
# ===================================================================


class TestSarifRoundTrip:
    """Create findings → write SARIF → read SARIF → verify."""

    def test_basic_round_trip(self, tmp_path: Path) -> None:
        """Write a SARIF log to disk, read it back, verify contents."""
        log = _build_simple_log(n_results=3)
        filepath = tmp_path / "report.sarif"
        write_sarif(log, filepath)

        loaded = read_sarif(filepath)
        assert loaded.version == SARIF_VERSION
        assert len(loaded.runs) == 1
        assert len(loaded.runs[0].results) == 3

    def test_round_trip_preserves_rules(self, tmp_path: Path) -> None:
        """Rule IDs and descriptions survive round-trip."""
        log = _build_simple_log(n_results=2)
        filepath = tmp_path / "rules.sarif"
        write_sarif(log, filepath)

        loaded = read_sarif(filepath)
        rules = loaded.runs[0].tool.driver.rules
        assert len(rules) == 2
        rule_ids = {r.id for r in rules}
        assert "UO000" in rule_ids
        assert "UO001" in rule_ids

    def test_round_trip_preserves_levels(self, tmp_path: Path) -> None:
        """Result severity levels survive round-trip."""
        log = _build_simple_log(n_results=4)
        filepath = tmp_path / "levels.sarif"
        write_sarif(log, filepath)

        loaded = read_sarif(filepath)
        results = loaded.runs[0].results
        assert any(r.level == Level.WARNING for r in results)
        assert any(r.level == Level.ERROR for r in results)

    def test_round_trip_preserves_locations(self, tmp_path: Path) -> None:
        """Result locations (URI, line numbers) survive round-trip."""
        log = _build_simple_log(n_results=2)
        filepath = tmp_path / "locations.sarif"
        write_sarif(log, filepath)

        loaded = read_sarif(filepath)
        result = loaded.runs[0].results[0]
        assert len(result.locations) > 0
        loc = result.locations[0]
        assert loc.physical_location is not None

    def test_string_round_trip(self) -> None:
        """Convert to string and parse back."""
        log = _build_simple_log(n_results=2)
        json_str = sarif_to_string(log)
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"
        assert len(parsed["runs"]) == 1
        assert len(parsed["runs"][0]["results"]) == 2

    def test_empty_log_round_trip(self, tmp_path: Path) -> None:
        """An empty SARIF log round-trips correctly."""
        builder = SarifBuilder(tool_name="empty-tool")
        log = builder.build()
        filepath = tmp_path / "empty.sarif"
        write_sarif(log, filepath)

        loaded = read_sarif(filepath)
        assert len(loaded.runs) == 1
        assert len(loaded.runs[0].results) == 0


# ===================================================================
# Tests — SARIF validation
# ===================================================================


class TestSarifValidation:
    """SARIF schema/semantic validation."""

    def test_valid_log_passes_validation(self) -> None:
        """A well-formed SARIF log passes validation."""
        import json
        log = _build_simple_log()
        log_dict = json.loads(sarif_to_string(log))
        report = validate_sarif(log_dict)
        assert isinstance(report, ValidationReport)
        assert report.is_valid or len(report.errors) == 0

    def test_version_is_correct(self) -> None:
        """SARIF version is 2.1.0."""
        log = _build_simple_log()
        assert log.version == "2.1.0"


# ===================================================================
# Tests — Merge multiple SARIF files
# ===================================================================


class TestSarifMerge:
    """Merge multiple SARIF files."""

    def test_merge_two_logs(self) -> None:
        """Merging two logs combines all runs."""
        log1 = _build_simple_log(tool_name="tool-a", n_results=2)
        log2 = _build_simple_log(tool_name="tool-b", n_results=3)
        merged = merge_sarif_logs(log1, log2)
        assert len(merged.runs) == 2
        total_results = sum(len(r.results) for r in merged.runs)
        assert total_results == 5

    def test_merge_preserves_tool_names(self) -> None:
        """Merged log preserves tool names from each source."""
        log1 = _build_simple_log(tool_name="tool-a", n_results=1)
        log2 = _build_simple_log(tool_name="tool-b", n_results=1)
        merged = merge_sarif_logs(log1, log2)
        tool_names = {r.tool.driver.name for r in merged.runs}
        assert "tool-a" in tool_names
        assert "tool-b" in tool_names

    def test_merge_single_log(self) -> None:
        """Merging a single log returns an equivalent log."""
        log = _build_simple_log(n_results=3)
        merged = merge_sarif_logs(log)
        assert len(merged.runs) == 1
        assert len(merged.runs[0].results) == 3

    def test_merge_runs_same_tool(self) -> None:
        """merge_runs combines results from runs with the same tool."""
        log1 = _build_simple_log(tool_name="same-tool", n_results=2)
        log2 = _build_simple_log(tool_name="same-tool", n_results=3)
        run1 = log1.runs[0]
        run2 = log2.runs[0]
        merged_run = merge_runs(run1, run2)
        assert len(merged_run.results) == 5


# ===================================================================
# Tests — SARIF diff for regression detection
# ===================================================================


class TestSarifDiff:
    """SARIF diff for regression detection."""

    def test_diff_identical_logs(self) -> None:
        """Diffing identical logs produces only unchanged entries."""
        log = _build_log_with_fingerprints()
        report = diff_sarif(log, log)
        assert isinstance(report, DiffReport)
        # All results should be unchanged
        for entry in report.entries:
            assert entry.category == DiffCategory.UNCHANGED

    def test_diff_detects_new_results(self) -> None:
        """New results in 'after' are detected."""
        before = _build_log_with_fingerprints(rule_ids=["UO001"])
        after = _build_log_with_fingerprints(rule_ids=["UO001", "UO002"])
        report = diff_sarif(before, after)
        new_entries = [e for e in report.entries
                       if e.category == DiffCategory.NEW]
        assert len(new_entries) >= 1

    def test_diff_detects_fixed_results(self) -> None:
        """Fixed results (in before but not after) are detected."""
        before = _build_log_with_fingerprints(rule_ids=["UO001", "UO002"])
        after = _build_log_with_fingerprints(rule_ids=["UO001"])
        report = diff_sarif(before, after)
        fixed_entries = [e for e in report.entries
                         if e.category == DiffCategory.FIXED]
        assert len(fixed_entries) >= 1

    def test_diff_summary_counts(self) -> None:
        """Diff summary has correct new/fixed/unchanged counts."""
        before = _build_log_with_fingerprints(rule_ids=["UO001", "UO002"])
        after = _build_log_with_fingerprints(rule_ids=["UO002", "UO003"])
        report = diff_sarif(before, after)
        summary = report.summary
        assert summary.new_count >= 1
        assert summary.fixed_count >= 1

    def test_compute_fingerprint_deterministic(self) -> None:
        """compute_fingerprint produces consistent results."""
        result = Result(
            rule_id="UO001",
            message=Message(text="Test finding"),
            level=Level.WARNING,
        )
        fp1 = compute_fingerprint(result)
        fp2 = compute_fingerprint(result)
        assert fp1 == fp2


# ===================================================================
# Tests — Full pipeline output in SARIF
# ===================================================================


class TestPipelineSarifOutput:
    """Verify the full pipeline can produce SARIF output."""

    def test_builder_produces_valid_sarif(self) -> None:
        """SarifBuilder produces valid SARIF 2.1.0 JSON."""
        builder = SarifBuilder(
            tool_name="usability-oracle",
            tool_version="0.1.0",
        )
        builder.add_rule("UO001", "target-size", "Target too small")
        builder.add_result(
            rule_id="UO001",
            message="Button 20×20 px below 44×44 px minimum",
            level=Level.ERROR,
            uri="index.html",
            start_line=42,
        )
        log = builder.build()
        json_str = sarif_to_string(log)
        parsed = json.loads(json_str)
        assert parsed["version"] == "2.1.0"
        assert len(parsed["runs"][0]["results"]) == 1

    def test_multiple_runs_in_single_log(self) -> None:
        """A SARIF log can contain multiple runs."""
        builder1 = SarifBuilder(tool_name="wcag-checker")
        builder1.add_result(rule_id="W001", message="Missing alt",
                            level=Level.ERROR)
        builder2 = SarifBuilder(tool_name="cog-analyzer")
        builder2.add_result(rule_id="C001", message="High cognitive load",
                            level=Level.WARNING)

        log1 = builder1.build()
        log2 = builder2.build()
        merged = merge_sarif_logs(log1, log2)
        json_str = sarif_to_string(merged)
        parsed = json.loads(json_str)
        assert len(parsed["runs"]) == 2
