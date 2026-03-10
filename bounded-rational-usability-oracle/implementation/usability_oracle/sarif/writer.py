"""
usability_oracle.sarif.writer — SARIF 2.1.0 file writer.

Generates SARIF 2.1.0 compliant JSON from schema objects or via a builder
pattern.  Supports pretty-print / compact modes and schema validation
before writing.

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, TextIO, Tuple, Union

from usability_oracle.sarif.schema import (
    SARIF_SCHEMA_URI,
    SARIF_VERSION,
    Artifact,
    ArtifactChange,
    ArtifactContent,
    ArtifactLocation,
    CodeFlow,
    Fix,
    Graph,
    Invocation,
    Level,
    Kind,
    Location,
    LogicalLocation,
    Message,
    MultiformatMessageString,
    Notification,
    PhysicalLocation,
    Region,
    Replacement,
    ReportingConfiguration,
    ReportingDescriptor,
    Result,
    ResultProvenance,
    Run,
    SarifLog,
    Suppression,
    ThreadFlow,
    ThreadFlowLocation,
    Tool,
    ToolComponent,
    ToolComponentReference,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Writer options
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class WriterOptions:
    """Configuration for SARIF output generation.

    Attributes:
        pretty: Emit human-readable indented JSON.
        indent: Indentation level when *pretty* is True.
        sort_keys: Alphabetically sort JSON keys.
        validate_before_write: Run validation before writing.
        ensure_ascii: Escape non-ASCII characters in output.
        omit_empty: Suppress keys with empty/default values.
    """

    pretty: bool = True
    indent: int = 2
    sort_keys: bool = False
    validate_before_write: bool = True
    ensure_ascii: bool = False
    omit_empty: bool = True


# ═══════════════════════════════════════════════════════════════════════════
# Builder
# ═══════════════════════════════════════════════════════════════════════════

class SarifBuilder:
    """Fluent builder for constructing SARIF documents.

    Usage::

        builder = SarifBuilder(tool_name="usability-oracle", tool_version="0.1.0")
        builder.add_rule("UO001", "target-size-minimum", "Target too small")
        builder.add_result(
            rule_id="UO001",
            message="Button size 20×20 px below minimum 44×44 px",
            level=Level.ERROR,
            uri="index.html",
            start_line=42,
        )
        log = builder.build()
    """

    def __init__(
        self,
        tool_name: str = "usability-oracle",
        tool_version: str = "0.1.0",
        tool_information_uri: str = "",
        tool_organization: str = "",
    ) -> None:
        self._tool_name = tool_name
        self._tool_version = tool_version
        self._tool_info_uri = tool_information_uri
        self._tool_org = tool_organization
        self._rules: List[ReportingDescriptor] = []
        self._rule_index: Dict[str, int] = {}
        self._results: List[Result] = []
        self._artifacts: List[Artifact] = []
        self._artifact_index: Dict[str, int] = {}
        self._invocations: List[Invocation] = []
        self._taxonomies: List[ToolComponent] = []
        self._extensions: List[ToolComponent] = []
        self._graphs: List[Graph] = []
        self._properties: Dict[str, Any] = {}
        self._original_uri_base_ids: Dict[str, ArtifactLocation] = {}
        self._automation_details: Optional[Dict[str, Any]] = None

    # --- Rules -----------------------------------------------------------

    def add_rule(
        self,
        rule_id: str,
        name: str = "",
        short_description: str = "",
        full_description: str = "",
        help_uri: str = "",
        default_level: Level = Level.WARNING,
        tags: Optional[List[str]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> SarifBuilder:
        """Register a reporting descriptor (rule)."""
        if rule_id in self._rule_index:
            return self  # Already registered.
        props = dict(properties or {})
        if tags:
            props["tags"] = tags
        rd = ReportingDescriptor(
            id=rule_id,
            name=name,
            short_description=(
                MultiformatMessageString(text=short_description)
                if short_description
                else None
            ),
            full_description=(
                MultiformatMessageString(text=full_description)
                if full_description
                else None
            ),
            help_uri=help_uri,
            default_configuration=ReportingConfiguration(level=default_level),
            properties=props,
        )
        self._rule_index[rule_id] = len(self._rules)
        self._rules.append(rd)
        return self

    # --- Results ---------------------------------------------------------

    def add_result(
        self,
        rule_id: str,
        message: str,
        level: Level = Level.WARNING,
        kind: Kind = Kind.FAIL,
        uri: str = "",
        start_line: Optional[int] = None,
        start_column: Optional[int] = None,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        logical_name: str = "",
        fingerprints: Optional[Dict[str, str]] = None,
        partial_fingerprints: Optional[Dict[str, str]] = None,
        fixes: Optional[Sequence[Fix]] = None,
        related_locations: Optional[Sequence[Location]] = None,
        code_flows: Optional[Sequence[CodeFlow]] = None,
        properties: Optional[Dict[str, Any]] = None,
    ) -> SarifBuilder:
        """Add a single result to the run."""
        # Auto-register the rule if not yet seen.
        if rule_id and rule_id not in self._rule_index:
            self.add_rule(rule_id)
        rule_idx = self._rule_index.get(rule_id, -1)

        # Build location.
        locations: List[Location] = []
        if uri or start_line is not None or logical_name:
            phys = None
            if uri or start_line is not None:
                region = None
                if start_line is not None:
                    region = Region(
                        start_line=start_line,
                        start_column=start_column,
                        end_line=end_line,
                        end_column=end_column,
                    )
                al = ArtifactLocation(uri=uri) if uri else None
                phys = PhysicalLocation(artifact_location=al, region=region)
            logical_locs: Tuple[LogicalLocation, ...] = ()
            if logical_name:
                logical_locs = (
                    LogicalLocation(name=logical_name, kind="element"),
                )
            locations.append(
                Location(
                    physical_location=phys,
                    logical_locations=logical_locs,
                )
            )

        # Ensure artifact is tracked.
        if uri and uri not in self._artifact_index:
            self._artifact_index[uri] = len(self._artifacts)
            self._artifacts.append(
                Artifact(location=ArtifactLocation(uri=uri))
            )

        result = Result(
            rule_id=rule_id,
            rule_index=rule_idx,
            level=level,
            kind=kind,
            message=Message(text=message),
            locations=tuple(locations),
            fingerprints=fingerprints or {},
            partial_fingerprints=partial_fingerprints or {},
            fixes=tuple(fixes) if fixes else (),
            related_locations=tuple(related_locations) if related_locations else (),
            code_flows=tuple(code_flows) if code_flows else (),
            properties=properties or {},
        )
        self._results.append(result)
        return self

    # --- Artifacts -------------------------------------------------------

    def add_artifact(
        self,
        uri: str,
        mime_type: str = "",
        roles: Optional[Sequence[str]] = None,
        content: str = "",
        description: str = "",
    ) -> SarifBuilder:
        """Register an artifact."""
        if uri in self._artifact_index:
            return self
        self._artifact_index[uri] = len(self._artifacts)
        art = Artifact(
            location=ArtifactLocation(uri=uri),
            description=Message(text=description) if description else None,
            mime_type=mime_type,
            roles=tuple(roles) if roles else (),
            contents=ArtifactContent(text=content) if content else None,
        )
        self._artifacts.append(art)
        return self

    # --- Invocations -----------------------------------------------------

    def set_invocation(
        self,
        command_line: str = "",
        execution_successful: bool = True,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None,
        working_directory: str = "",
        exit_code: int = 0,
    ) -> SarifBuilder:
        """Set the invocation record for the run."""
        wd = ArtifactLocation(uri=working_directory) if working_directory else None
        inv = Invocation(
            command_line=command_line,
            execution_successful=execution_successful,
            start_time_utc=start_time or "",
            end_time_utc=end_time or "",
            working_directory=wd,
            exit_code=exit_code,
        )
        self._invocations = [inv]
        return self

    # --- Taxonomies ------------------------------------------------------

    def add_taxonomy(self, taxonomy: ToolComponent) -> SarifBuilder:
        """Attach an external taxonomy to the run."""
        self._taxonomies.append(taxonomy)
        return self

    # --- Extensions ------------------------------------------------------

    def add_extension(self, extension: ToolComponent) -> SarifBuilder:
        """Add a tool extension."""
        self._extensions.append(extension)
        return self

    # --- Automation ------------------------------------------------------

    def set_automation(
        self,
        automation_id: str,
        guid: str = "",
        correlation_guid: str = "",
    ) -> SarifBuilder:
        """Set automation details."""
        self._automation_details = {
            "id": automation_id,
        }
        if guid:
            self._automation_details["guid"] = guid
        if correlation_guid:
            self._automation_details["correlationGuid"] = correlation_guid
        return self

    # --- URI bases -------------------------------------------------------

    def add_uri_base(self, base_id: str, uri: str) -> SarifBuilder:
        """Register an originalUriBaseId."""
        self._original_uri_base_ids[base_id] = ArtifactLocation(uri=uri)
        return self

    # --- Properties ------------------------------------------------------

    def set_property(self, key: str, value: Any) -> SarifBuilder:
        """Set a run-level property."""
        self._properties[key] = value
        return self

    # --- Build -----------------------------------------------------------

    def build(self) -> SarifLog:
        """Construct the final :class:`SarifLog`."""
        driver = ToolComponent(
            name=self._tool_name,
            version=self._tool_version,
            information_uri=self._tool_info_uri,
            organization=self._tool_org,
            rules=tuple(self._rules),
        )
        tool = Tool(driver=driver, extensions=tuple(self._extensions))

        run = Run(
            tool=tool,
            invocations=tuple(self._invocations),
            artifacts=tuple(self._artifacts),
            results=tuple(self._results),
            taxonomies=tuple(self._taxonomies),
            original_uri_base_ids=self._original_uri_base_ids,
            automation_details=self._automation_details,
            graphs=tuple(self._graphs),
            properties=self._properties,
        )
        return SarifLog(runs=(run,))


# ═══════════════════════════════════════════════════════════════════════════
# Writer
# ═══════════════════════════════════════════════════════════════════════════

class SarifWriter:
    """Serialize a :class:`SarifLog` to JSON.

    Usage::

        writer = SarifWriter()
        writer.write_file(log, "output.sarif")
        # or
        json_str = writer.to_string(log)
    """

    def __init__(self, options: Optional[WriterOptions] = None) -> None:
        self.options = options or WriterOptions()

    def write_file(
        self,
        log: SarifLog,
        path: Union[str, Path],
    ) -> None:
        """Write a SARIF log to a file."""
        text = self.to_string(log)
        Path(path).write_text(text, encoding="utf-8")

    def write_stream(self, log: SarifLog, stream: TextIO) -> None:
        """Write a SARIF log to a text stream."""
        stream.write(self.to_string(log))

    def to_string(self, log: SarifLog) -> str:
        """Serialize a SARIF log to a JSON string."""
        if self.options.validate_before_write:
            errors = log.validate()
            if errors:
                logger.warning("SARIF validation errors: %s", errors)

        d = log.to_dict()
        if self.options.omit_empty:
            d = _strip_empty(d)

        indent = self.options.indent if self.options.pretty else None
        return json.dumps(
            d,
            indent=indent,
            sort_keys=self.options.sort_keys,
            ensure_ascii=self.options.ensure_ascii,
        )

    def to_dict(self, log: SarifLog) -> Dict[str, Any]:
        """Serialize to dict (with optional validation)."""
        if self.options.validate_before_write:
            errors = log.validate()
            if errors:
                logger.warning("SARIF validation errors: %s", errors)
        d = log.to_dict()
        if self.options.omit_empty:
            d = _strip_empty(d)
        return d


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

def _strip_empty(obj: Any) -> Any:
    """Recursively remove keys with empty/null values from dicts."""
    if isinstance(obj, dict):
        return {
            k: _strip_empty(v)
            for k, v in obj.items()
            if v is not None and v != "" and v != [] and v != {}
        }
    if isinstance(obj, list):
        return [_strip_empty(item) for item in obj]
    return obj


# ═══════════════════════════════════════════════════════════════════════════
# Convenience functions
# ═══════════════════════════════════════════════════════════════════════════

def write_sarif(
    log: SarifLog,
    path: Union[str, Path],
    *,
    pretty: bool = True,
    validate: bool = True,
) -> None:
    """Write a SARIF log to a file (convenience function)."""
    writer = SarifWriter(
        WriterOptions(pretty=pretty, validate_before_write=validate)
    )
    writer.write_file(log, path)


def sarif_to_string(
    log: SarifLog,
    *,
    pretty: bool = True,
    validate: bool = True,
) -> str:
    """Serialize a SARIF log to a JSON string (convenience function)."""
    writer = SarifWriter(
        WriterOptions(pretty=pretty, validate_before_write=validate)
    )
    return writer.to_string(log)
