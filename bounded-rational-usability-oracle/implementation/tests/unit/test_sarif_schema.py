"""Unit tests for usability_oracle.sarif.schema.

Tests round-trip serialization (to_dict / from_dict), validation, and
optional field handling for all major SARIF 2.1.0 schema types.
"""

from __future__ import annotations

import pytest

from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Artifact,
    ArtifactChange,
    ArtifactContent,
    ArtifactLocation,
    BaselineState,
    Fix,
    Kind,
    Level,
    Location,
    LogicalLocation,
    Message,
    MultiformatMessageString,
    Notification,
    NotificationLevel,
    PhysicalLocation,
    Region,
    Replacement,
    ReportingConfiguration,
    ReportingDescriptor,
    Result,
    Run,
    SarifLog,
    Suppression,
    SuppressionKind,
    SuppressionStatus,
    Tool,
    ToolComponent,
    ToolComponentReference,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_location(uri: str = "file.py", line: int = 10) -> Location:
    return Location(
        physical_location=PhysicalLocation(
            artifact_location=ArtifactLocation(uri=uri),
            region=Region(start_line=line),
        ),
    )


def _make_result(
    rule_id: str = "R001",
    message: str = "msg",
    level: Level = Level.ERROR,
) -> Result:
    return Result(
        rule_id=rule_id,
        message=Message(text=message),
        level=level,
        kind=Kind.FAIL,
        locations=(_make_location(),),
        fingerprints={},
        partial_fingerprints={},
    )


def _make_run(results: tuple[Result, ...] = ()) -> Run:
    return Run(
        tool=Tool(driver=ToolComponent(name="test-tool")),
        results=results,
    )


def _make_log(runs: tuple[Run, ...] | None = None) -> SarifLog:
    if runs is None:
        runs = (_make_run((_make_result(),)),)
    return SarifLog(
        version=SARIF_VERSION,
        schema_uri=SARIF_SCHEMA_URI,
        runs=runs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Message
# ═══════════════════════════════════════════════════════════════════════════


class TestMessage:
    """Tests for Message round-trip and from-string parsing."""

    def test_to_dict_from_dict_round_trip(self) -> None:
        msg = Message(text="hello", markdown="**hello**")
        d = msg.to_dict()
        restored = Message.from_dict(d)
        assert restored.text == "hello"
        assert restored.markdown == "**hello**"

    def test_from_string(self) -> None:
        msg = Message.from_dict("plain text")
        assert msg.text == "plain text"

    def test_empty_dict(self) -> None:
        msg = Message.from_dict({})
        assert msg.text == ""

    def test_arguments_round_trip(self) -> None:
        msg = Message(text="{0} found", arguments=("bug",))
        d = msg.to_dict()
        assert d["arguments"] == ["bug"]
        restored = Message.from_dict(d)
        assert restored.arguments == ("bug",)


# ═══════════════════════════════════════════════════════════════════════════
# MultiformatMessageString
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiformatMessageString:
    """Tests for MultiformatMessageString."""

    def test_round_trip(self) -> None:
        mms = MultiformatMessageString(text="desc", markdown="**desc**")
        d = mms.to_dict()
        restored = MultiformatMessageString.from_dict(d)
        assert restored.text == "desc"
        assert restored.markdown == "**desc**"

    def test_from_string(self) -> None:
        mms = MultiformatMessageString.from_dict("short")
        assert mms.text == "short"


# ═══════════════════════════════════════════════════════════════════════════
# ArtifactLocation
# ═══════════════════════════════════════════════════════════════════════════


class TestArtifactLocation:
    """Tests for ArtifactLocation with uri, uriBaseId, index."""

    def test_uri_round_trip(self) -> None:
        al = ArtifactLocation(uri="src/main.py")
        d = al.to_dict()
        assert d["uri"] == "src/main.py"
        restored = ArtifactLocation.from_dict(d)
        assert restored.uri == "src/main.py"

    def test_uri_base_id(self) -> None:
        al = ArtifactLocation(uri="main.py", uri_base_id="SRCROOT")
        d = al.to_dict()
        assert d["uriBaseId"] == "SRCROOT"
        restored = ArtifactLocation.from_dict(d)
        assert restored.uri_base_id == "SRCROOT"

    def test_index(self) -> None:
        al = ArtifactLocation(uri="f.py", index=3)
        d = al.to_dict()
        assert d["index"] == 3
        restored = ArtifactLocation.from_dict(d)
        assert restored.index == 3

    def test_negative_index_omitted(self) -> None:
        al = ArtifactLocation(uri="f.py")
        d = al.to_dict()
        assert "index" not in d


# ═══════════════════════════════════════════════════════════════════════════
# Region
# ═══════════════════════════════════════════════════════════════════════════


class TestRegion:
    """Tests for Region with startLine/startColumn and validation."""

    def test_start_line_column_round_trip(self) -> None:
        r = Region(start_line=5, start_column=10, end_line=5, end_column=20)
        d = r.to_dict()
        assert d["startLine"] == 5
        assert d["startColumn"] == 10
        restored = Region.from_dict(d)
        assert restored.start_line == 5
        assert restored.end_column == 20

    def test_validate_negative_start_line(self) -> None:
        r = Region(start_line=0)
        errors = r.validate()
        assert any("startLine" in e for e in errors)

    def test_validate_negative_start_column(self) -> None:
        r = Region(start_column=0)
        errors = r.validate()
        assert any("startColumn" in e for e in errors)

    def test_validate_end_before_start(self) -> None:
        r = Region(start_line=10, end_line=5)
        errors = r.validate()
        assert any("endLine" in e for e in errors)

    def test_validate_valid_region(self) -> None:
        r = Region(start_line=1, start_column=1)
        assert r.validate() == []

    def test_optional_fields_omitted(self) -> None:
        r = Region(start_line=1)
        d = r.to_dict()
        assert "startColumn" not in d
        assert "endLine" not in d

    def test_char_offset_validation(self) -> None:
        r = Region(char_offset=-1)
        errors = r.validate()
        assert any("charOffset" in e for e in errors)

    def test_byte_offset_validation(self) -> None:
        r = Region(byte_offset=-1)
        errors = r.validate()
        assert any("byteOffset" in e for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# PhysicalLocation and Location
# ═══════════════════════════════════════════════════════════════════════════


class TestPhysicalLocationAndLocation:
    """Tests for PhysicalLocation and Location nested round-trip."""

    def test_physical_location_round_trip(self) -> None:
        pl = PhysicalLocation(
            artifact_location=ArtifactLocation(uri="a.py"),
            region=Region(start_line=1),
        )
        d = pl.to_dict()
        restored = PhysicalLocation.from_dict(d)
        assert restored.artifact_location is not None
        assert restored.artifact_location.uri == "a.py"
        assert restored.region is not None
        assert restored.region.start_line == 1

    def test_location_with_logical(self) -> None:
        loc = Location(
            physical_location=PhysicalLocation(
                artifact_location=ArtifactLocation(uri="b.py"),
            ),
            logical_locations=(
                LogicalLocation(name="MyClass.method", kind="function"),
            ),
        )
        d = loc.to_dict()
        restored = Location.from_dict(d)
        assert len(restored.logical_locations) == 1
        assert restored.logical_locations[0].name == "MyClass.method"

    def test_location_empty(self) -> None:
        loc = Location()
        d = loc.to_dict()
        assert "physicalLocation" not in d


# ═══════════════════════════════════════════════════════════════════════════
# Artifact
# ═══════════════════════════════════════════════════════════════════════════


class TestArtifact:
    """Tests for Artifact with location, mimeType, roles."""

    def test_round_trip(self) -> None:
        art = Artifact(
            location=ArtifactLocation(uri="src/app.js"),
            mime_type="application/javascript",
            roles=("analysisTarget",),
        )
        d = art.to_dict()
        assert d["mimeType"] == "application/javascript"
        assert d["roles"] == ["analysisTarget"]
        restored = Artifact.from_dict(d)
        assert restored.mime_type == "application/javascript"
        assert restored.roles == ("analysisTarget",)

    def test_optional_fields(self) -> None:
        art = Artifact(location=ArtifactLocation(uri="f.py"))
        d = art.to_dict()
        assert "mimeType" not in d
        assert "roles" not in d


# ═══════════════════════════════════════════════════════════════════════════
# ReportingDescriptor
# ═══════════════════════════════════════════════════════════════════════════


class TestReportingDescriptor:
    """Tests for ReportingDescriptor (rule)."""

    def test_round_trip(self) -> None:
        rd = ReportingDescriptor(
            id="R001",
            name="no-eval",
            short_description=MultiformatMessageString(text="Avoid eval"),
            full_description=MultiformatMessageString(text="Do not use eval()"),
        )
        d = rd.to_dict()
        assert d["id"] == "R001"
        assert d["shortDescription"]["text"] == "Avoid eval"
        restored = ReportingDescriptor.from_dict(d)
        assert restored.id == "R001"
        assert restored.name == "no-eval"
        assert restored.short_description is not None
        assert restored.short_description.text == "Avoid eval"

    def test_validate_missing_id(self) -> None:
        rd = ReportingDescriptor(id="")
        assert len(rd.validate()) > 0

    def test_validate_with_id(self) -> None:
        rd = ReportingDescriptor(id="R1")
        assert rd.validate() == []


# ═══════════════════════════════════════════════════════════════════════════
# Tool and ToolComponent
# ═══════════════════════════════════════════════════════════════════════════


class TestToolAndToolComponent:
    """Tests for Tool and ToolComponent round-trip."""

    def test_tool_component_round_trip(self) -> None:
        tc = ToolComponent(
            name="my-tool",
            version="1.0.0",
            rules=(ReportingDescriptor(id="R1", name="rule-one"),),
        )
        d = tc.to_dict()
        assert d["name"] == "my-tool"
        assert len(d["rules"]) == 1
        restored = ToolComponent.from_dict(d)
        assert restored.name == "my-tool"
        assert len(restored.rules) == 1

    def test_tool_round_trip(self) -> None:
        tool = Tool(driver=ToolComponent(name="driver"))
        d = tool.to_dict()
        restored = Tool.from_dict(d)
        assert restored.driver.name == "driver"

    def test_tool_component_validate_missing_name(self) -> None:
        tc = ToolComponent(name="")
        assert len(tc.validate()) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════


class TestResult:
    """Tests for Result with all fields and from_dict."""

    def test_round_trip(self) -> None:
        result = _make_result()
        d = result.to_dict()
        assert d["ruleId"] == "R001"
        assert d["level"] == "error"
        assert d["kind"] == "fail"
        restored = Result.from_dict(d)
        assert restored.rule_id == "R001"
        assert restored.level == Level.ERROR
        assert restored.message.text == "msg"

    def test_from_dict_defaults(self) -> None:
        d = {"message": {"text": "hi"}}
        r = Result.from_dict(d)
        assert r.level == Level.WARNING
        assert r.kind == Kind.FAIL
        assert r.rule_id == ""

    def test_locations_preserved(self) -> None:
        result = _make_result()
        d = result.to_dict()
        restored = Result.from_dict(d)
        assert len(restored.locations) == 1
        pl = restored.locations[0].physical_location
        assert pl is not None
        assert pl.artifact_location is not None
        assert pl.artifact_location.uri == "file.py"

    def test_fingerprints_round_trip(self) -> None:
        result = Result(
            rule_id="R1",
            message=Message(text="m"),
            fingerprints={"v1": "abc123"},
            partial_fingerprints={"v1": "partial"},
        )
        d = result.to_dict()
        assert d["fingerprints"] == {"v1": "abc123"}
        restored = Result.from_dict(d)
        assert restored.fingerprints == {"v1": "abc123"}
        assert restored.partial_fingerprints == {"v1": "partial"}

    def test_optional_fields_omitted(self) -> None:
        result = Result(rule_id="R1", message=Message(text="m"))
        d = result.to_dict()
        assert "fingerprints" not in d
        assert "suppressions" not in d
        assert "fixes" not in d


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════


class TestRun:
    """Tests for Run with tool, results, artifacts."""

    def test_round_trip(self) -> None:
        run = _make_run((_make_result(),))
        d = run.to_dict()
        assert d["tool"]["driver"]["name"] == "test-tool"
        assert len(d["results"]) == 1
        restored = Run.from_dict(d)
        assert restored.tool.driver.name == "test-tool"
        assert len(restored.results) == 1

    def test_num_results(self) -> None:
        run = _make_run((_make_result(), _make_result()))
        assert run.num_results == 2

    def test_num_errors(self) -> None:
        run = _make_run((
            _make_result(level=Level.ERROR),
            _make_result(level=Level.WARNING),
        ))
        assert run.num_errors == 1
        assert run.num_warnings == 1

    def test_with_artifacts(self) -> None:
        run = Run(
            tool=Tool(driver=ToolComponent(name="t")),
            artifacts=(Artifact(location=ArtifactLocation(uri="a.py")),),
        )
        d = run.to_dict()
        assert len(d["artifacts"]) == 1
        restored = Run.from_dict(d)
        assert len(restored.artifacts) == 1


# ═══════════════════════════════════════════════════════════════════════════
# SarifLog
# ═══════════════════════════════════════════════════════════════════════════


class TestSarifLog:
    """Tests for SarifLog round-trip, validate(), total_results."""

    def test_round_trip(self) -> None:
        log = _make_log()
        d = log.to_dict()
        assert d["version"] == SARIF_VERSION
        assert d["$schema"] == SARIF_SCHEMA_URI
        restored = SarifLog.from_dict(d)
        assert restored.version == SARIF_VERSION
        assert len(restored.runs) == 1

    def test_total_results(self) -> None:
        r1 = _make_result(rule_id="A")
        r2 = _make_result(rule_id="B")
        log = _make_log((_make_run((r1, r2)),))
        assert log.total_results == 2

    def test_total_errors(self) -> None:
        log = _make_log((_make_run((_make_result(level=Level.ERROR),)),))
        assert log.total_errors == 1

    def test_validate_valid(self) -> None:
        log = _make_log()
        assert log.validate() == []

    def test_validate_no_runs(self) -> None:
        log = SarifLog(runs=())
        errors = log.validate()
        assert len(errors) > 0

    def test_validate_bad_version(self) -> None:
        log = SarifLog(version="1.0.0", runs=(_make_run(),))
        errors = log.validate()
        assert any("version" in e.lower() for e in errors)


# ═══════════════════════════════════════════════════════════════════════════
# Fix, Replacement, ArtifactChange
# ═══════════════════════════════════════════════════════════════════════════


class TestFixReplacementArtifactChange:
    """Tests for Fix, Replacement, ArtifactChange."""

    def test_replacement_round_trip(self) -> None:
        rep = Replacement(
            deleted_region=Region(start_line=5, start_column=1, end_line=5, end_column=10),
            inserted_content=ArtifactContent(text="new_code"),
        )
        d = rep.to_dict()
        restored = Replacement.from_dict(d)
        assert restored.deleted_region.start_line == 5
        assert restored.inserted_content is not None
        assert restored.inserted_content.text == "new_code"

    def test_artifact_change_round_trip(self) -> None:
        ac = ArtifactChange(
            artifact_location=ArtifactLocation(uri="f.py"),
            replacements=(
                Replacement(deleted_region=Region(start_line=1)),
            ),
        )
        d = ac.to_dict()
        restored = ArtifactChange.from_dict(d)
        assert restored.artifact_location.uri == "f.py"
        assert len(restored.replacements) == 1

    def test_fix_round_trip(self) -> None:
        fix = Fix(
            description=Message(text="Apply fix"),
            artifact_changes=(
                ArtifactChange(
                    artifact_location=ArtifactLocation(uri="x.py"),
                    replacements=(
                        Replacement(
                            deleted_region=Region(start_line=1, end_line=1),
                            inserted_content=ArtifactContent(text="fixed"),
                        ),
                    ),
                ),
            ),
        )
        d = fix.to_dict()
        restored = Fix.from_dict(d)
        assert restored.description.text == "Apply fix"
        assert len(restored.artifact_changes) == 1


# ═══════════════════════════════════════════════════════════════════════════
# Suppression
# ═══════════════════════════════════════════════════════════════════════════


class TestSuppression:
    """Tests for Suppression round-trip."""

    def test_round_trip(self) -> None:
        s = Suppression(
            kind=SuppressionKind.EXTERNAL,
            justification="wontfix",
            status=SuppressionStatus.ACCEPTED,
        )
        d = s.to_dict()
        assert d["kind"] == "external"
        assert d["justification"] == "wontfix"
        assert d["status"] == "accepted"
        restored = Suppression.from_dict(d)
        assert restored.kind == SuppressionKind.EXTERNAL
        assert restored.status == SuppressionStatus.ACCEPTED

    def test_default_kind(self) -> None:
        s = Suppression()
        assert s.kind == SuppressionKind.IN_SOURCE


# ═══════════════════════════════════════════════════════════════════════════
# Optional field handling
# ═══════════════════════════════════════════════════════════════════════════


class TestOptionalFieldHandling:
    """Fields not set should not appear in the dict output."""

    def test_message_empty_markdown_omitted(self) -> None:
        msg = Message(text="only text")
        d = msg.to_dict()
        assert "markdown" not in d

    def test_artifact_location_empty_uri_base_id_omitted(self) -> None:
        al = ArtifactLocation(uri="f.py")
        d = al.to_dict()
        assert "uriBaseId" not in d

    def test_result_empty_collections_omitted(self) -> None:
        r = Result(rule_id="R1", message=Message(text="m"))
        d = r.to_dict()
        assert "fixes" not in d
        assert "codeFlows" not in d
        assert "suppressions" not in d

    def test_run_empty_artifacts_omitted(self) -> None:
        run = _make_run()
        d = run.to_dict()
        assert "artifacts" not in d


# ═══════════════════════════════════════════════════════════════════════════
# Level and Kind enum values
# ═══════════════════════════════════════════════════════════════════════════


class TestEnumValues:
    """Tests for Level and Kind enum values."""

    def test_level_values(self) -> None:
        assert Level.NONE.value == "none"
        assert Level.NOTE.value == "note"
        assert Level.WARNING.value == "warning"
        assert Level.ERROR.value == "error"

    def test_kind_values(self) -> None:
        assert Kind.PASS.value == "pass"
        assert Kind.FAIL.value == "fail"
        assert Kind.OPEN.value == "open"
        assert Kind.INFORMATIONAL.value == "informational"
        assert Kind.NOT_APPLICABLE.value == "notApplicable"
        assert Kind.REVIEW.value == "review"

    def test_baseline_state_values(self) -> None:
        assert BaselineState.NEW.value == "new"
        assert BaselineState.UNCHANGED.value == "unchanged"
        assert BaselineState.UPDATED.value == "updated"
        assert BaselineState.ABSENT.value == "absent"
