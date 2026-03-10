"""Unit tests for usability_oracle.sarif.validator.

Tests SarifValidator for valid documents, missing/invalid fields,
cross-reference checks, and ValidationReport properties.
"""

from __future__ import annotations

from typing import Any, Dict

import pytest

from usability_oracle.sarif.schema import SARIF_SCHEMA_URI, SARIF_VERSION
from usability_oracle.sarif.validator import (
    SarifValidator,
    ValidationError,
    ValidationReport,
    ValidationSeverity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _valid_sarif() -> Dict[str, Any]:
    """Return a minimal valid SARIF 2.1.0 document dict."""
    return {
        "version": SARIF_VERSION,
        "$schema": SARIF_SCHEMA_URI,
        "runs": [{
            "tool": {
                "driver": {
                    "name": "test-tool",
                    "rules": [{"id": "R1", "shortDescription": {"text": "rule"}}],
                },
            },
            "results": [{
                "ruleId": "R1",
                "ruleIndex": 0,
                "message": {"text": "Found issue"},
                "level": "error",
            }],
        }],
    }


def _valid_sarif_no_results() -> Dict[str, Any]:
    return {
        "version": SARIF_VERSION,
        "$schema": SARIF_SCHEMA_URI,
        "runs": [{
            "tool": {"driver": {"name": "test-tool"}},
            "results": [],
        }],
    }


# ═══════════════════════════════════════════════════════════════════════════
# Valid document
# ═══════════════════════════════════════════════════════════════════════════


class TestValidDocument:
    """Tests that a valid document passes validation."""

    def test_valid_passes(self) -> None:
        v = SarifValidator()
        report = v.validate(_valid_sarif())
        assert report.is_valid is True
        assert report.error_count == 0

    def test_valid_no_results(self) -> None:
        v = SarifValidator()
        report = v.validate(_valid_sarif_no_results())
        assert report.is_valid is True


# ═══════════════════════════════════════════════════════════════════════════
# Missing top-level fields
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingTopLevelFields:
    """Tests for missing required top-level fields."""

    def test_missing_version(self) -> None:
        data = _valid_sarif()
        del data["version"]
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("version" in e.message.lower() for e in report.errors)

    def test_missing_runs(self) -> None:
        data = _valid_sarif()
        del data["runs"]
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("runs" in e.message.lower() for e in report.errors)

    def test_invalid_version(self) -> None:
        data = _valid_sarif()
        data["version"] = "9.9.9"
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0


# ═══════════════════════════════════════════════════════════════════════════
# Missing tool / driver
# ═══════════════════════════════════════════════════════════════════════════


class TestMissingToolDriver:
    """Tests for missing tool and driver fields in a run."""

    def test_missing_tool(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [{"results": []}],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("tool" in e.message.lower() for e in report.errors)

    def test_missing_driver(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [{"tool": {}, "results": []}],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("driver" in e.message.lower() for e in report.errors)

    def test_missing_driver_name(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [{"tool": {"driver": {}}, "results": []}],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("name" in e.message.lower() for e in report.errors)


# ═══════════════════════════════════════════════════════════════════════════
# Empty runs array
# ═══════════════════════════════════════════════════════════════════════════


class TestEmptyRuns:
    """Tests that empty runs array generates a warning."""

    def test_empty_runs_warning(self) -> None:
        data = {"version": SARIF_VERSION, "$schema": SARIF_SCHEMA_URI, "runs": []}
        v = SarifValidator()
        report = v.validate(data)
        assert report.warning_count > 0

    def test_empty_runs_still_valid(self) -> None:
        data = {"version": SARIF_VERSION, "$schema": SARIF_SCHEMA_URI, "runs": []}
        v = SarifValidator()
        report = v.validate(data)
        assert report.is_valid is True


# ═══════════════════════════════════════════════════════════════════════════
# Invalid ruleIndex cross-reference
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidRuleIndex:
    """Tests for ruleIndex out of bounds."""

    def test_rule_index_out_of_bounds(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [{
                "tool": {
                    "driver": {
                        "name": "t",
                        "rules": [{"id": "R1"}],
                    },
                },
                "results": [{
                    "ruleIndex": 5,
                    "message": {"text": "m"},
                }],
            }],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("ruleIndex" in e.message or "rule" in e.message.lower()
                    for e in report.errors)

    def test_valid_rule_index(self) -> None:
        report = SarifValidator().validate(_valid_sarif())
        # No error about ruleIndex
        rule_idx_errors = [
            e for e in report.errors
            if "ruleIndex" in e.json_path and e.severity == ValidationSeverity.ERROR
        ]
        assert len(rule_idx_errors) == 0


# ═══════════════════════════════════════════════════════════════════════════
# ValidationReport
# ═══════════════════════════════════════════════════════════════════════════


class TestValidationReport:
    """Tests for ValidationReport properties."""

    def test_is_valid_with_no_errors(self) -> None:
        report = ValidationReport()
        assert report.is_valid is True
        assert report.error_count == 0
        assert report.warning_count == 0

    def test_is_valid_with_error(self) -> None:
        report = ValidationReport()
        report.add("$.version", "missing", severity=ValidationSeverity.ERROR)
        assert report.is_valid is False
        assert report.error_count == 1

    def test_warning_does_not_invalidate(self) -> None:
        report = ValidationReport()
        report.add("$.runs", "empty", severity=ValidationSeverity.WARNING)
        assert report.is_valid is True
        assert report.warning_count == 1

    def test_mixed_errors_and_warnings(self) -> None:
        report = ValidationReport()
        report.add("$.version", "bad", severity=ValidationSeverity.ERROR)
        report.add("$.runs", "empty", severity=ValidationSeverity.WARNING)
        report.add("$.runs[0]", "bad tool", severity=ValidationSeverity.ERROR)
        assert report.error_count == 2
        assert report.warning_count == 1
        assert report.is_valid is False

    def test_summary_string(self) -> None:
        report = ValidationReport()
        report.add("$", "e", severity=ValidationSeverity.ERROR)
        s = report.summary()
        assert "1 error" in s


# ═══════════════════════════════════════════════════════════════════════════
# URI format validation
# ═══════════════════════════════════════════════════════════════════════════


class TestURIValidation:
    """Tests that URI fields are checked for format."""

    def test_invalid_schema_uri_warning(self) -> None:
        data = _valid_sarif()
        data["$schema"] = ":::not-a-uri"
        v = SarifValidator()
        report = v.validate(data)
        # Should produce a warning for invalid URI
        uri_issues = [
            e for e in report.errors
            if "$schema" in e.json_path.lower() or "schema" in e.message.lower()
        ]
        assert len(uri_issues) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Non-dict root
# ═══════════════════════════════════════════════════════════════════════════


class TestNonDictRoot:
    """Tests that a non-dict root produces an error."""

    def test_list_root(self) -> None:
        v = SarifValidator()
        report = v.validate([1, 2, 3])
        assert report.error_count > 0
        assert report.is_valid is False

    def test_string_root(self) -> None:
        v = SarifValidator()
        report = v.validate("not a dict")
        assert report.error_count > 0

    def test_none_root(self) -> None:
        v = SarifValidator()
        report = v.validate(None)
        assert report.error_count > 0


# ═══════════════════════════════════════════════════════════════════════════
# Nested run validation
# ═══════════════════════════════════════════════════════════════════════════


class TestNestedRunValidation:
    """Tests for validation within individual runs."""

    def test_non_dict_run(self) -> None:
        data = {"version": SARIF_VERSION, "runs": ["not-a-dict"]}
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0

    def test_multiple_runs_validated(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [
                {"tool": {"driver": {"name": "ok"}}, "results": []},
                {"results": []},  # missing tool
            ],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("tool" in e.message.lower() for e in report.errors)

    def test_result_missing_message(self) -> None:
        data = {
            "version": SARIF_VERSION,
            "runs": [{
                "tool": {"driver": {"name": "t"}},
                "results": [{"ruleId": "R1"}],
            }],
        }
        v = SarifValidator()
        report = v.validate(data)
        assert report.error_count > 0
        assert any("message" in e.message.lower() for e in report.errors)
