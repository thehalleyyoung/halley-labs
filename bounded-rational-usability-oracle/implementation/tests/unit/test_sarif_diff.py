"""Unit tests for usability_oracle.sarif.diff.

Tests fingerprinting, SarifDiffer, DiffSummary, and DiffReport for
various before/after scenarios.
"""

from __future__ import annotations

import pytest

from usability_oracle.sarif.diff import (
    DiffCategory,
    DiffEntry,
    DiffReport,
    DiffSummary,
    SarifDiffer,
    compute_fingerprint,
    compute_partial_fingerprint,
)
from usability_oracle.sarif.schema import (
    ArtifactLocation,
    Level,
    Location,
    LogicalLocation,
    Message,
    PhysicalLocation,
    Region,
    Result,
    Run,
    SarifLog,
    Tool,
    ToolComponent,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _result(
    rule_id: str = "R1",
    uri: str = "file.py",
    line: int = 10,
    message: str = "msg",
    level: Level = Level.ERROR,
    fingerprints: dict[str, str] | None = None,
    logical_name: str = "",
) -> Result:
    locs: list[LogicalLocation] = []
    if logical_name:
        locs.append(LogicalLocation(name=logical_name, kind="function"))
    return Result(
        rule_id=rule_id,
        message=Message(text=message),
        level=level,
        locations=(
            Location(
                physical_location=PhysicalLocation(
                    artifact_location=ArtifactLocation(uri=uri),
                    region=Region(start_line=line),
                ),
                logical_locations=tuple(locs),
            ),
        ),
        fingerprints=fingerprints or {},
    )


def _log(*results: Result) -> SarifLog:
    run = Run(
        tool=Tool(driver=ToolComponent(name="test")),
        results=tuple(results),
    )
    return SarifLog(runs=(run,))


# ═══════════════════════════════════════════════════════════════════════════
# compute_fingerprint
# ═══════════════════════════════════════════════════════════════════════════


class TestComputeFingerprint:
    """Tests for compute_fingerprint determinism and content."""

    def test_deterministic(self) -> None:
        r = _result()
        fp1 = compute_fingerprint(r)
        fp2 = compute_fingerprint(r)
        assert fp1 == fp2

    def test_uses_rule_id_and_location(self) -> None:
        r1 = _result(rule_id="A", uri="a.py", line=1)
        r2 = _result(rule_id="A", uri="a.py", line=2)
        assert compute_fingerprint(r1) != compute_fingerprint(r2)

    def test_different_rule_ids(self) -> None:
        r1 = _result(rule_id="A")
        r2 = _result(rule_id="B")
        assert compute_fingerprint(r1) != compute_fingerprint(r2)

    def test_existing_fingerprint_used(self) -> None:
        r = _result(fingerprints={"v1": "custom-fp-123"})
        assert compute_fingerprint(r) == "custom-fp-123"

    def test_returns_string(self) -> None:
        r = _result()
        fp = compute_fingerprint(r)
        assert isinstance(fp, str)
        assert len(fp) > 0


# ═══════════════════════════════════════════════════════════════════════════
# compute_partial_fingerprint
# ═══════════════════════════════════════════════════════════════════════════


class TestComputePartialFingerprint:
    """Tests for compute_partial_fingerprint."""

    def test_uses_rule_id_and_logical_location(self) -> None:
        r1 = _result(rule_id="R1", logical_name="foo")
        r2 = _result(rule_id="R1", logical_name="bar")
        assert compute_partial_fingerprint(r1) != compute_partial_fingerprint(r2)

    def test_same_rule_same_logical_same_fp(self) -> None:
        r1 = _result(rule_id="R1", uri="a.py", line=1, logical_name="fn")
        r2 = _result(rule_id="R1", uri="b.py", line=99, logical_name="fn")
        assert compute_partial_fingerprint(r1) == compute_partial_fingerprint(r2)

    def test_deterministic(self) -> None:
        r = _result(logical_name="method")
        pfp1 = compute_partial_fingerprint(r)
        pfp2 = compute_partial_fingerprint(r)
        assert pfp1 == pfp2


# ═══════════════════════════════════════════════════════════════════════════
# SarifDiffer.diff — new / fixed / unchanged / updated
# ═══════════════════════════════════════════════════════════════════════════


class TestSarifDifferDiff:
    """Tests for SarifDiffer.diff with various result changes."""

    def test_identical_logs_all_unchanged(self) -> None:
        r = _result()
        before = _log(r)
        after = _log(r)
        report = SarifDiffer().diff(before, after)
        assert report.summary.unchanged_count == 1
        assert report.summary.new_count == 0
        assert report.summary.fixed_count == 0

    def test_new_result_detected(self) -> None:
        before = _log()
        after = _log(_result(rule_id="NEW", uri="new.py", line=1))
        report = SarifDiffer().diff(before, after)
        assert report.summary.new_count == 1
        assert report.summary.fixed_count == 0

    def test_fixed_result_detected(self) -> None:
        before = _log(_result(rule_id="OLD", uri="old.py", line=1))
        after = _log()
        report = SarifDiffer().diff(before, after)
        assert report.summary.fixed_count == 1
        assert report.summary.new_count == 0

    def test_updated_result_detected(self) -> None:
        r_before = _result(rule_id="R1", message="old msg")
        r_after = _result(rule_id="R1", message="new msg")
        before = _log(r_before)
        after = _log(r_after)
        report = SarifDiffer().diff(before, after)
        assert report.summary.updated_count == 1

    def test_empty_before_all_new(self) -> None:
        before = _log()
        after = _log(
            _result(rule_id="A", uri="a.py", line=1),
            _result(rule_id="B", uri="b.py", line=2),
        )
        report = SarifDiffer().diff(before, after)
        assert report.summary.new_count == 2
        assert report.summary.total_after == 2

    def test_empty_after_all_fixed(self) -> None:
        before = _log(
            _result(rule_id="A", uri="a.py", line=1),
            _result(rule_id="B", uri="b.py", line=2),
        )
        after = _log()
        report = SarifDiffer().diff(before, after)
        assert report.summary.fixed_count == 2
        assert report.summary.total_before == 2


# ═══════════════════════════════════════════════════════════════════════════
# DiffSummary
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffSummary:
    """Tests for DiffSummary counts and properties."""

    def test_net_change(self) -> None:
        s = DiffSummary(new_count=3, fixed_count=1)
        assert s.net_change == 2

    def test_net_change_negative(self) -> None:
        s = DiffSummary(new_count=0, fixed_count=5)
        assert s.net_change == -5

    def test_is_regression(self) -> None:
        assert DiffSummary(new_count=1).is_regression is True
        assert DiffSummary(new_count=0).is_regression is False

    def test_is_improvement(self) -> None:
        assert DiffSummary(fixed_count=2, new_count=0).is_improvement is True
        assert DiffSummary(fixed_count=2, new_count=1).is_improvement is False
        assert DiffSummary(fixed_count=0, new_count=0).is_improvement is False

    def test_to_dict(self) -> None:
        s = DiffSummary(
            total_before=10,
            total_after=12,
            new_count=3,
            fixed_count=1,
            unchanged_count=8,
            updated_count=1,
        )
        d = s.to_dict()
        assert d["totalBefore"] == 10
        assert d["totalAfter"] == 12
        assert d["new"] == 3
        assert d["fixed"] == 1
        assert d["netChange"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# DiffReport
# ═══════════════════════════════════════════════════════════════════════════


class TestDiffReport:
    """Tests for DiffReport property accessors."""

    def test_new_results_property(self) -> None:
        r_new = _result(rule_id="NEW")
        before = _log()
        after = _log(r_new)
        report = SarifDiffer().diff(before, after)
        assert len(report.new_results) == 1

    def test_fixed_results_property(self) -> None:
        r_old = _result(rule_id="OLD")
        before = _log(r_old)
        after = _log()
        report = SarifDiffer().diff(before, after)
        assert len(report.fixed_results) == 1

    def test_unchanged_results_property(self) -> None:
        r = _result()
        report = SarifDiffer().diff(_log(r), _log(r))
        assert len(report.unchanged_results) == 1

    def test_updated_results_property(self) -> None:
        r_before = _result(rule_id="R1", message="a")
        r_after = _result(rule_id="R1", message="b")
        report = SarifDiffer().diff(_log(r_before), _log(r_after))
        updated = report.updated_results
        assert len(updated) == 1
        assert updated[0].previous is not None

    def test_empty_report(self) -> None:
        report = DiffReport()
        assert len(report.entries) == 0
        assert report.summary.net_change == 0


# ═══════════════════════════════════════════════════════════════════════════
# Partial fingerprint fallback matching
# ═══════════════════════════════════════════════════════════════════════════


class TestPartialFingerprintFallback:
    """Tests for partial fingerprint matching when lines shift."""

    def test_line_shift_matched_by_partial(self) -> None:
        r_before = _result(rule_id="R1", uri="f.py", line=10, logical_name="fn")
        r_after = _result(rule_id="R1", uri="f.py", line=15, logical_name="fn")
        differ = SarifDiffer(use_partial_fingerprints=True)
        report = differ.diff(_log(r_before), _log(r_after))
        assert report.summary.updated_count == 1
        assert report.summary.new_count == 0
        assert report.summary.fixed_count == 0

    def test_no_partial_fingerprint_treats_as_new_and_fixed(self) -> None:
        r_before = _result(rule_id="R1", uri="f.py", line=10, logical_name="fn")
        r_after = _result(rule_id="R1", uri="f.py", line=15, logical_name="fn")
        differ = SarifDiffer(use_partial_fingerprints=False)
        report = differ.diff(_log(r_before), _log(r_after))
        assert report.summary.new_count == 1
        assert report.summary.fixed_count == 1
