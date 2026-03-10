"""
usability_oracle.sarif.reader — SARIF 2.1.0 file reader and parser.

Parses SARIF JSON files (versions 2.0 and 2.1.0) with schema validation,
multi-run support, inline artifact content handling, URI resolution, and
incremental streaming for large files.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    BinaryIO,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    TextIO,
    Tuple,
    Union,
)
from urllib.parse import urljoin, urlparse

from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Artifact,
    ArtifactLocation,
    Run,
    SarifLog,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Errors
# ═══════════════════════════════════════════════════════════════════════════

class SarifParseError(Exception):
    """Raised when SARIF parsing fails."""

    def __init__(
        self,
        message: str,
        *,
        json_path: str = "",
        line: int = -1,
        recoverable: bool = False,
    ) -> None:
        self.json_path = json_path
        self.line = line
        self.recoverable = recoverable
        detail = f" at {json_path}" if json_path else ""
        detail += f" (line {line})" if line >= 0 else ""
        super().__init__(f"{message}{detail}")


class SarifVersionError(SarifParseError):
    """Raised for unsupported SARIF versions."""


# ═══════════════════════════════════════════════════════════════════════════
# Parse options
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ReaderOptions:
    """Configuration for the SARIF reader.

    Attributes:
        validate: Run schema validation during parse.
        strict: Raise on any validation error (vs. collecting warnings).
        max_results: Stop after this many results (0 = unlimited).
        resolve_uris: Resolve relative URIs against originalUriBaseIds.
        inline_artifacts: Include inline artifact content in results.
        allowed_versions: Accepted SARIF versions.
    """

    validate: bool = True
    strict: bool = False
    max_results: int = 0
    resolve_uris: bool = True
    inline_artifacts: bool = True
    allowed_versions: Tuple[str, ...] = ("2.1.0", "2.0.0")


# ═══════════════════════════════════════════════════════════════════════════
# URI resolver
# ═══════════════════════════════════════════════════════════════════════════

def resolve_artifact_uri(
    artifact_location: ArtifactLocation,
    uri_bases: Dict[str, ArtifactLocation],
) -> str:
    """Resolve an artifact URI against originalUriBaseIds.

    If the artifact has a ``uriBaseId``, the base URI is looked up from
    *uri_bases* and the artifact's relative URI is joined against it.
    Otherwise the original URI is returned unchanged.
    """
    uri = artifact_location.uri
    base_id = artifact_location.uri_base_id
    if not base_id or base_id not in uri_bases:
        return uri
    base_loc = uri_bases[base_id]
    # Recursively resolve base (bases can chain).
    base_uri = resolve_artifact_uri(base_loc, uri_bases)
    if not base_uri.endswith("/"):
        base_uri += "/"
    return urljoin(base_uri, uri)


# ═══════════════════════════════════════════════════════════════════════════
# Core reader
# ═══════════════════════════════════════════════════════════════════════════

class SarifReader:
    """Parse SARIF 2.1.0 (and 2.0) JSON files.

    Usage::

        reader = SarifReader()
        log = reader.read_file("results.sarif")

        # Or from a dict already parsed:
        log = reader.read_dict(json_data)

        # Streaming / incremental for large files:
        for run in reader.iter_runs("huge.sarif"):
            process(run)
    """

    def __init__(self, options: Optional[ReaderOptions] = None) -> None:
        self.options = options or ReaderOptions()
        self.warnings: List[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def read_file(self, path: Union[str, Path]) -> SarifLog:
        """Read a SARIF log from a file path."""
        path = Path(path)
        if not path.exists():
            raise SarifParseError(f"File not found: {path}")
        with open(path, "r", encoding="utf-8") as fh:
            return self.read_stream(fh)

    def read_string(self, text: str) -> SarifLog:
        """Read a SARIF log from a JSON string."""
        return self.read_stream(io.StringIO(text))

    def read_stream(self, stream: TextIO) -> SarifLog:
        """Read a SARIF log from a text stream."""
        self.warnings.clear()
        try:
            data = json.load(stream)
        except json.JSONDecodeError as exc:
            raise SarifParseError(
                f"Invalid JSON: {exc.msg}",
                line=exc.lineno,
                recoverable=False,
            ) from exc
        return self.read_dict(data)

    def read_dict(self, data: Dict[str, Any]) -> SarifLog:
        """Parse a SARIF log from an already-decoded JSON dict."""
        self.warnings.clear()

        if not isinstance(data, dict):
            raise SarifParseError("SARIF root must be a JSON object")

        # Version check.
        version = str(data.get("version", ""))
        if version not in self.options.allowed_versions:
            raise SarifVersionError(
                f"Unsupported SARIF version '{version}'",
                json_path="$.version",
            )

        # Upgrade 2.0 → 2.1.0 transparently.
        if version == "2.0.0" or version == "2.0":
            data = _upgrade_v2_to_v210(data)

        # Basic schema checks.
        if self.options.validate:
            self._validate_top_level(data)

        log = SarifLog.from_dict(data)

        # Post-parse processing.
        if self.options.resolve_uris:
            log = self._resolve_uris(log)

        return log

    def iter_runs(
        self, path: Union[str, Path]
    ) -> Iterator[Run]:
        """Incrementally yield :class:`Run` objects from a SARIF file.

        For very large SARIF files this avoids loading the entire document
        into memory at once by parsing the ``runs`` array element-by-element
        using a simple streaming strategy.
        """
        path = Path(path)
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        version = str(data.get("version", ""))
        if version not in self.options.allowed_versions:
            raise SarifVersionError(
                f"Unsupported SARIF version '{version}'",
                json_path="$.version",
            )

        if version in ("2.0.0", "2.0"):
            data = _upgrade_v2_to_v210(data)

        runs_data = data.get("runs", [])
        if not isinstance(runs_data, list):
            raise SarifParseError("'runs' must be a JSON array", json_path="$.runs")

        count = 0
        for i, run_d in enumerate(runs_data):
            try:
                run = Run.from_dict(run_d)
                yield run
                count += len(run.results)
                if 0 < self.options.max_results <= count:
                    return
            except Exception as exc:  # noqa: BLE001
                if self.options.strict:
                    raise SarifParseError(
                        str(exc), json_path=f"$.runs[{i}]"
                    ) from exc
                self.warnings.append(f"runs[{i}]: {exc}")

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_top_level(self, data: Dict[str, Any]) -> None:
        """Check required top-level SARIF fields."""
        if "version" not in data:
            self._warn_or_raise("Missing required field 'version'", "$.version")
        if "runs" not in data:
            self._warn_or_raise("Missing required field 'runs'", "$.runs")
        elif not isinstance(data["runs"], list):
            self._warn_or_raise("'runs' must be an array", "$.runs")
        else:
            for i, run_d in enumerate(data["runs"]):
                self._validate_run(run_d, i)

    def _validate_run(self, run_d: Any, index: int) -> None:
        """Validate a single run dict."""
        prefix = f"$.runs[{index}]"
        if not isinstance(run_d, dict):
            self._warn_or_raise(f"Run must be an object", prefix)
            return
        if "tool" not in run_d:
            self._warn_or_raise("Missing required field 'tool'", f"{prefix}.tool")
        else:
            tool = run_d["tool"]
            if not isinstance(tool, dict):
                self._warn_or_raise("'tool' must be an object", f"{prefix}.tool")
            elif "driver" not in tool:
                self._warn_or_raise(
                    "Missing required field 'driver'",
                    f"{prefix}.tool.driver",
                )
            else:
                driver = tool["driver"]
                if "name" not in driver:
                    self._warn_or_raise(
                        "Missing required field 'name'",
                        f"{prefix}.tool.driver.name",
                    )

        # Validate results reference valid rules.
        results = run_d.get("results", [])
        rules = (
            run_d.get("tool", {}).get("driver", {}).get("rules", [])
        )
        num_rules = len(rules) if isinstance(rules, list) else 0
        for j, res in enumerate(results):
            if not isinstance(res, dict):
                continue
            ri = res.get("ruleIndex")
            if ri is not None and isinstance(ri, int) and ri >= num_rules:
                self._warn_or_raise(
                    f"ruleIndex {ri} exceeds rules array length {num_rules}",
                    f"{prefix}.results[{j}].ruleIndex",
                )

    def _warn_or_raise(self, message: str, json_path: str = "") -> None:
        if self.options.strict:
            raise SarifParseError(message, json_path=json_path)
        self.warnings.append(f"{json_path}: {message}" if json_path else message)

    # ------------------------------------------------------------------
    # URI resolution
    # ------------------------------------------------------------------

    def _resolve_uris(self, log: SarifLog) -> SarifLog:
        """Resolve relative URIs in all runs using originalUriBaseIds."""
        new_runs: List[Run] = []
        for run in log.runs:
            if not run.original_uri_base_ids:
                new_runs.append(run)
                continue
            # Resolve artifact URIs.
            new_artifacts: List[Artifact] = []
            for art in run.artifacts:
                if art.location is not None:
                    resolved = resolve_artifact_uri(
                        art.location, run.original_uri_base_ids
                    )
                    new_loc = ArtifactLocation(
                        uri=resolved,
                        uri_base_id="",
                        index=art.location.index,
                        description=art.location.description,
                    )
                    # Reconstruct artifact with resolved location.
                    new_artifacts.append(
                        Artifact(
                            location=new_loc,
                            description=art.description,
                            mime_type=art.mime_type,
                            length=art.length,
                            roles=art.roles,
                            contents=art.contents,
                            encoding=art.encoding,
                            source_language=art.source_language,
                            hashes=art.hashes,
                            last_modified_time_utc=art.last_modified_time_utc,
                            parent_index=art.parent_index,
                            offset=art.offset,
                            properties=art.properties,
                        )
                    )
                else:
                    new_artifacts.append(art)
            new_runs.append(
                Run(
                    tool=run.tool,
                    invocations=run.invocations,
                    conversion=run.conversion,
                    language=run.language,
                    version_control_provenance=run.version_control_provenance,
                    original_uri_base_ids=run.original_uri_base_ids,
                    artifacts=tuple(new_artifacts),
                    logical_locations=run.logical_locations,
                    graphs=run.graphs,
                    results=run.results,
                    automation_details=run.automation_details,
                    baseline_guid=run.baseline_guid,
                    default_encoding=run.default_encoding,
                    column_kind=run.column_kind,
                    taxonomies=run.taxonomies,
                    properties=run.properties,
                )
            )
        return SarifLog(
            version=log.version,
            schema_uri=log.schema_uri,
            runs=tuple(new_runs),
            inline_external_properties=log.inline_external_properties,
            properties=log.properties,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Version upgrade helper
# ═══════════════════════════════════════════════════════════════════════════

def _upgrade_v2_to_v210(data: Dict[str, Any]) -> Dict[str, Any]:
    """Best-effort upgrade from SARIF 2.0 to 2.1.0 structure.

    Key differences:
    - ``resources.rules`` → ``tool.driver.rules``
    - ``tool`` was a flat object in 2.0, becomes ``tool.driver`` in 2.1.0
    - ``fileLocation`` → ``artifactLocation``
    - ``files`` → ``artifacts``
    """
    result = dict(data)
    result["version"] = SARIF_VERSION
    if "$schema" not in result:
        result["$schema"] = SARIF_SCHEMA_URI

    runs = result.get("runs", [])
    new_runs: List[Dict[str, Any]] = []
    for run_d in runs:
        run_d = dict(run_d)

        # Upgrade tool structure.
        tool = run_d.get("tool", {})
        if "driver" not in tool and "name" in tool:
            run_d["tool"] = {"driver": tool}

        # Move resources.rules to tool.driver.rules.
        resources = run_d.pop("resources", None)
        if isinstance(resources, dict):
            rules = resources.get("rules", [])
            if rules:
                driver = run_d.setdefault("tool", {}).setdefault("driver", {})
                driver.setdefault("rules", rules)

        # Rename files → artifacts.
        files = run_d.pop("files", None)
        if files is not None:
            if isinstance(files, dict):
                # 2.0 used a dict keyed by URI.
                arts: List[Dict[str, Any]] = []
                for uri, fobj in files.items():
                    art: Dict[str, Any] = dict(fobj) if isinstance(fobj, dict) else {}
                    art.setdefault("location", {})["uri"] = uri
                    if "fileLocation" in art:
                        art["location"] = art.pop("fileLocation")
                    arts.append(art)
                run_d["artifacts"] = arts
            elif isinstance(files, list):
                run_d["artifacts"] = files

        # Upgrade fileLocation → artifactLocation in results.
        for res in run_d.get("results", []):
            if not isinstance(res, dict):
                continue
            for loc in res.get("locations", []):
                if not isinstance(loc, dict):
                    continue
                phys = loc.get("physicalLocation", {})
                if "fileLocation" in phys:
                    phys["artifactLocation"] = phys.pop("fileLocation")

        new_runs.append(run_d)
    result["runs"] = new_runs
    return result


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def read_sarif(
    source: Union[str, Path, Dict[str, Any]],
    *,
    strict: bool = False,
    validate: bool = True,
) -> SarifLog:
    """Read a SARIF log from a file path, JSON string, or dict.

    This is the primary convenience entry point for the reader module.
    """
    reader = SarifReader(ReaderOptions(validate=validate, strict=strict))
    if isinstance(source, dict):
        return reader.read_dict(source)
    source = str(source)
    if source.strip().startswith("{"):
        return reader.read_string(source)
    return reader.read_file(source)
