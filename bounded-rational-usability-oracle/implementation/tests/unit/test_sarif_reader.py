"""Unit tests for usability_oracle.sarif.reader.

Tests SarifReader for valid/invalid input, version handling, multi-run
files, streaming, options, and error conditions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

from usability_oracle.sarif.reader import (
    ReaderOptions,
    SarifParseError,
    SarifReader,
    SarifVersionError,
)
from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Run,
    SarifLog,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _minimal_sarif(
    version: str = SARIF_VERSION,
    runs: list[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """Build a minimal valid SARIF dict."""
    if runs is None:
        runs = [{"tool": {"driver": {"name": "test"}}, "results": []}]
    return {"version": version, "$schema": SARIF_SCHEMA_URI, "runs": runs}


def _sarif_with_result(
    rule_id: str = "R1",
    message: str = "msg",
    level: str = "error",
) -> Dict[str, Any]:
    return _minimal_sarif(runs=[{
        "tool": {
            "driver": {
                "name": "test",
                "rules": [{"id": rule_id, "shortDescription": {"text": rule_id}}],
            }
        },
        "results": [{
            "ruleId": rule_id,
            "ruleIndex": 0,
            "message": {"text": message},
            "level": level,
        }],
    }])


# ═══════════════════════════════════════════════════════════════════════════
# read_dict — valid input
# ═══════════════════════════════════════════════════════════════════════════


class TestReadDictValid:
    """Tests for read_dict with valid SARIF 2.1.0 data."""

    def test_minimal_sarif(self) -> None:
        reader = SarifReader()
        log = reader.read_dict(_minimal_sarif())
        assert isinstance(log, SarifLog)
        assert log.version == SARIF_VERSION
        assert len(log.runs) == 1

    def test_sarif_with_result(self) -> None:
        reader = SarifReader()
        log = reader.read_dict(_sarif_with_result())
        assert log.total_results == 1
        assert log.runs[0].results[0].rule_id == "R1"

    def test_tool_name_preserved(self) -> None:
        reader = SarifReader()
        log = reader.read_dict(_minimal_sarif())
        assert log.runs[0].tool.driver.name == "test"


# ═══════════════════════════════════════════════════════════════════════════
# read_string
# ═══════════════════════════════════════════════════════════════════════════


class TestReadString:
    """Tests for read_string with JSON string input."""

    def test_valid_json_string(self) -> None:
        reader = SarifReader()
        text = json.dumps(_minimal_sarif())
        log = reader.read_string(text)
        assert isinstance(log, SarifLog)
        assert len(log.runs) == 1

    def test_string_with_results(self) -> None:
        reader = SarifReader()
        text = json.dumps(_sarif_with_result())
        log = reader.read_string(text)
        assert log.total_results == 1


# ═══════════════════════════════════════════════════════════════════════════
# Version handling
# ═══════════════════════════════════════════════════════════════════════════


class TestVersionHandling:
    """Tests for SARIF version detection and upgrade."""

    def test_version_2_1_0(self) -> None:
        reader = SarifReader()
        log = reader.read_dict(_minimal_sarif("2.1.0"))
        assert log.version == "2.1.0"

    def test_version_2_0_0_upgraded(self) -> None:
        data = {
            "version": "2.0.0",
            "runs": [{
                "tool": {"name": "old-tool"},
                "results": [],
            }],
        }
        reader = SarifReader()
        log = reader.read_dict(data)
        assert log.version == SARIF_VERSION

    def test_unsupported_version_raises(self) -> None:
        reader = SarifReader()
        with pytest.raises(SarifVersionError):
            reader.read_dict({"version": "3.0.0", "runs": []})

    def test_missing_version_raises(self) -> None:
        reader = SarifReader()
        with pytest.raises((SarifVersionError, SarifParseError)):
            reader.read_dict({"runs": []})


# ═══════════════════════════════════════════════════════════════════════════
# Multi-run files
# ═══════════════════════════════════════════════════════════════════════════


class TestMultiRun:
    """Tests for SARIF files with multiple runs."""

    def test_two_runs(self) -> None:
        runs = [
            {"tool": {"driver": {"name": "tool-a"}}, "results": []},
            {"tool": {"driver": {"name": "tool-b"}}, "results": []},
        ]
        reader = SarifReader()
        log = reader.read_dict(_minimal_sarif(runs=runs))
        assert len(log.runs) == 2
        assert log.runs[0].tool.driver.name == "tool-a"
        assert log.runs[1].tool.driver.name == "tool-b"


# ═══════════════════════════════════════════════════════════════════════════
# Invalid input
# ═══════════════════════════════════════════════════════════════════════════


class TestInvalidInput:
    """Tests for error handling on invalid input."""

    def test_invalid_json_raises(self) -> None:
        reader = SarifReader()
        with pytest.raises(SarifParseError):
            reader.read_string("{not valid json")

    def test_non_dict_root_raises(self) -> None:
        reader = SarifReader()
        with pytest.raises(SarifParseError):
            reader.read_dict([1, 2, 3])  # type: ignore[arg-type]

    def test_null_root_raises(self) -> None:
        reader = SarifReader()
        with pytest.raises(SarifParseError):
            reader.read_dict(None)  # type: ignore[arg-type]


# ═══════════════════════════════════════════════════════════════════════════
# ReaderOptions
# ═══════════════════════════════════════════════════════════════════════════


class TestReaderOptions:
    """Tests for ReaderOptions: strict mode, max_results."""

    def test_default_options(self) -> None:
        opts = ReaderOptions()
        assert opts.strict is False
        assert opts.max_results == 0
        assert opts.validate is True

    def test_strict_mode_raises_on_warnings(self) -> None:
        data = _minimal_sarif(runs=[{
            "tool": {"driver": {"name": "t"}},
            "results": [{"ruleIndex": 99, "message": {"text": "m"}}],
        }])
        reader = SarifReader(ReaderOptions(strict=True))
        with pytest.raises(SarifParseError):
            reader.read_dict(data)

    def test_non_strict_collects_warnings(self) -> None:
        data = _minimal_sarif(runs=[{
            "tool": {"driver": {"name": "t"}},
            "results": [{"ruleIndex": 99, "message": {"text": "m"}}],
        }])
        reader = SarifReader(ReaderOptions(strict=False))
        log = reader.read_dict(data)
        assert isinstance(log, SarifLog)
        assert len(reader.warnings) > 0


# ═══════════════════════════════════════════════════════════════════════════
# iter_runs
# ═══════════════════════════════════════════════════════════════════════════


class TestIterRuns:
    """Tests for iter_runs yielding Run objects."""

    def test_iter_runs_from_file(self, tmp_path: Path) -> None:
        data = _minimal_sarif(runs=[
            {"tool": {"driver": {"name": "t1"}}, "results": []},
            {"tool": {"driver": {"name": "t2"}}, "results": []},
        ])
        path = tmp_path / "test.sarif"
        path.write_text(json.dumps(data), encoding="utf-8")

        reader = SarifReader()
        runs = list(reader.iter_runs(path))
        assert len(runs) == 2
        assert all(isinstance(r, Run) for r in runs)
        assert runs[0].tool.driver.name == "t1"

    def test_iter_runs_max_results(self, tmp_path: Path) -> None:
        results = [{"message": {"text": f"r{i}"}} for i in range(10)]
        data = _minimal_sarif(runs=[{
            "tool": {"driver": {"name": "t"}},
            "results": results,
        }])
        path = tmp_path / "big.sarif"
        path.write_text(json.dumps(data), encoding="utf-8")

        reader = SarifReader(ReaderOptions(max_results=5, validate=False))
        runs = list(reader.iter_runs(path))
        assert len(runs) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# Warnings collected
# ═══════════════════════════════════════════════════════════════════════════


class TestWarnings:
    """Tests for warning collection in non-strict mode."""

    def test_missing_tool_driver_name_warning(self) -> None:
        data = _minimal_sarif(runs=[{
            "tool": {"driver": {}},
            "results": [],
        }])
        reader = SarifReader(ReaderOptions(strict=False))
        reader.read_dict(data)
        assert len(reader.warnings) > 0

    def test_warnings_cleared_on_new_read(self) -> None:
        data_bad = _minimal_sarif(runs=[{
            "tool": {"driver": {}},
            "results": [],
        }])
        data_good = _minimal_sarif()
        reader = SarifReader(ReaderOptions(strict=False))
        reader.read_dict(data_bad)
        assert len(reader.warnings) > 0
        reader.read_dict(data_good)
        assert len(reader.warnings) == 0
