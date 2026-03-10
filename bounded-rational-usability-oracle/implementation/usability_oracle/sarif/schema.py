"""
usability_oracle.sarif.schema — Complete SARIF 2.1.0 object model.

Frozen dataclasses modelling the full Static Analysis Results Interchange
Format (SARIF) Version 2.1.0 (OASIS Standard, 27 March 2020).

Reference: https://docs.oasis-open.org/sarif/sarif/v2.1.0/sarif-v2.1.0.html

Every type provides ``to_dict()`` / ``from_dict()`` for JSON round-tripping
and a ``validate()`` method for schema-constraint checking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any, Dict, List, Optional, Sequence, Tuple


# ═══════════════════════════════════════════════════════════════════════════
# Enumerations
# ═══════════════════════════════════════════════════════════════════════════

SARIF_VERSION = "2.1.0"
SARIF_SCHEMA_URI = (
    "https://docs.oasis-open.org/sarif/sarif/v2.1.0/"
    "cos02/schemas/sarif-schema-2.1.0.json"
)


@unique
class Level(Enum):
    """SARIF result level (§3.27.10)."""
    NONE = "none"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


@unique
class Kind(Enum):
    """SARIF result kind (§3.27.9)."""
    PASS = "pass"
    OPEN = "open"
    INFORMATIONAL = "informational"
    NOT_APPLICABLE = "notApplicable"
    REVIEW = "review"
    FAIL = "fail"


@unique
class BaselineState(Enum):
    """SARIF result baseline state (§3.27.24)."""
    NEW = "new"
    UNCHANGED = "unchanged"
    UPDATED = "updated"
    ABSENT = "absent"


@unique
class SuppressionKind(Enum):
    """Suppression kind (§3.35.2)."""
    IN_SOURCE = "inSource"
    EXTERNAL = "external"


@unique
class SuppressionStatus(Enum):
    """Suppression status (§3.35.4)."""
    ACCEPTED = "accepted"
    UNDER_REVIEW = "underReview"
    REJECTED = "rejected"


@unique
class ThreadFlowImportance(Enum):
    """Thread-flow location importance (§3.38.13)."""
    IMPORTANT = "important"
    ESSENTIAL = "essential"
    UNIMPORTANT = "unimportant"


@unique
class RoleEnum(Enum):
    """Artifact role (§3.24.6)."""
    ANALYSIS_TARGET = "analysisTarget"
    ATTACHMENT = "attachment"
    RESPONSE_FILE = "responseFile"
    RESULT_FILE = "resultFile"
    STANDARD_STREAM = "standardStream"
    TRACED_FILE = "tracedFile"
    UNMODIFIED = "unmodified"
    MODIFIED = "modified"
    ADDED = "added"
    DELETED = "deleted"
    RENAMED = "renamed"
    UNCONTROLLED = "uncontrolled"
    DRIVER = "driver"
    EXTENSION = "extension"
    TRANSLATION = "translation"
    TAXONOMY = "taxonomy"
    POLICY = "policy"
    REFERENCED_ON_COMMAND_LINE = "referencedOnCommandLine"
    MEMORY_CONTENTS = "memoryContents"
    DIRECTORY = "directory"
    USER_SPECIFIED_CONFIGURATION = "userSpecifiedConfiguration"
    TOOL_SPECIFIED_CONFIGURATION = "toolSpecifiedConfiguration"
    DEBUG_OUTPUT_FILE = "debugOutputFile"


@unique
class NotificationLevel(Enum):
    """Notification level (§3.58.6)."""
    NONE = "none"
    NOTE = "note"
    WARNING = "warning"
    ERROR = "error"


# ═══════════════════════════════════════════════════════════════════════════
# Helper
# ═══════════════════════════════════════════════════════════════════════════

def _opt(d: Dict[str, Any], key: str, val: Any) -> None:
    """Set *key* in *d* only when *val* is truthy / non-default."""
    if val is not None and val != "" and val != () and val != {} and val != []:
        d[key] = val


def _enum_val(e: Optional[Enum]) -> Optional[str]:
    return e.value if e is not None else None


# ═══════════════════════════════════════════════════════════════════════════
# Message types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class MultiformatMessageString:
    """A message string with optional Markdown (§3.12)."""
    text: str = ""
    markdown: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"text": self.text}
        _opt(d, "markdown", self.markdown)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Any) -> MultiformatMessageString:
        if isinstance(d, str):
            return cls(text=d)
        if not isinstance(d, dict):
            return cls()
        return cls(
            text=str(d.get("text", "")),
            markdown=str(d.get("markdown", "")),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Message:
    """SARIF message object (§3.11)."""
    text: str = ""
    markdown: str = ""
    id: str = ""
    arguments: Tuple[str, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "text", self.text)
        _opt(d, "markdown", self.markdown)
        _opt(d, "id", self.id)
        if self.arguments:
            d["arguments"] = list(self.arguments)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Any) -> Message:
        if isinstance(d, str):
            return cls(text=d)
        if not isinstance(d, dict):
            return cls()
        return cls(
            text=str(d.get("text", "")),
            markdown=str(d.get("markdown", "")),
            id=str(d.get("id", "")),
            arguments=tuple(d.get("arguments", [])),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Location types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ArtifactLocation:
    """Reference to an artifact by URI (§3.4)."""
    uri: str = ""
    uri_base_id: str = ""
    index: int = -1
    description: Message = field(default_factory=Message)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "uri", self.uri)
        _opt(d, "uriBaseId", self.uri_base_id)
        if self.index >= 0:
            d["index"] = self.index
        if self.description.text:
            d["description"] = self.description.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactLocation:
        return cls(
            uri=str(d.get("uri", "")),
            uri_base_id=str(d.get("uriBaseId", "")),
            index=int(d.get("index", -1)),
            description=Message.from_dict(d.get("description", {})),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Region:
    """A region within an artifact (§3.30)."""
    start_line: Optional[int] = None
    start_column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    char_offset: Optional[int] = None
    char_length: Optional[int] = None
    byte_offset: Optional[int] = None
    byte_length: Optional[int] = None
    snippet: Optional[ArtifactContent] = None
    message: Optional[Message] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.start_line is not None:
            d["startLine"] = self.start_line
        if self.start_column is not None:
            d["startColumn"] = self.start_column
        if self.end_line is not None:
            d["endLine"] = self.end_line
        if self.end_column is not None:
            d["endColumn"] = self.end_column
        if self.char_offset is not None:
            d["charOffset"] = self.char_offset
        if self.char_length is not None:
            d["charLength"] = self.char_length
        if self.byte_offset is not None:
            d["byteOffset"] = self.byte_offset
        if self.byte_length is not None:
            d["byteLength"] = self.byte_length
        if self.snippet is not None:
            d["snippet"] = self.snippet.to_dict()
        if self.message is not None:
            d["message"] = self.message.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Region:
        snippet = d.get("snippet")
        msg = d.get("message")
        return cls(
            start_line=d.get("startLine"),
            start_column=d.get("startColumn"),
            end_line=d.get("endLine"),
            end_column=d.get("endColumn"),
            char_offset=d.get("charOffset"),
            char_length=d.get("charLength"),
            byte_offset=d.get("byteOffset"),
            byte_length=d.get("byteLength"),
            snippet=ArtifactContent.from_dict(snippet) if snippet else None,
            message=Message.from_dict(msg) if msg else None,
            properties=d.get("properties", {}),
        )

    def validate(self) -> List[str]:
        """Return a list of validation errors, empty if valid."""
        errors: List[str] = []
        if self.start_line is not None and self.start_line < 1:
            errors.append("startLine must be >= 1")
        if self.start_column is not None and self.start_column < 1:
            errors.append("startColumn must be >= 1")
        if self.end_line is not None and self.start_line is not None:
            if self.end_line < self.start_line:
                errors.append("endLine must be >= startLine")
        if self.char_offset is not None and self.char_offset < 0:
            errors.append("charOffset must be >= 0")
        if self.byte_offset is not None and self.byte_offset < 0:
            errors.append("byteOffset must be >= 0")
        return errors


@dataclass(frozen=True, slots=True)
class ArtifactContent:
    """Artifact content — inline text, binary, or rendered (§3.3)."""
    text: str = ""
    binary: str = ""
    rendered: Optional[MultiformatMessageString] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "text", self.text)
        _opt(d, "binary", self.binary)
        if self.rendered is not None:
            d["rendered"] = self.rendered.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactContent:
        rendered = d.get("rendered")
        return cls(
            text=str(d.get("text", "")),
            binary=str(d.get("binary", "")),
            rendered=(
                MultiformatMessageString.from_dict(rendered) if rendered else None
            ),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class LogicalLocation:
    """A logical location (namespace, class, function) (§3.33)."""
    name: str = ""
    index: int = -1
    fully_qualified_name: str = ""
    decorated_name: str = ""
    parent_index: int = -1
    kind: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "name", self.name)
        if self.index >= 0:
            d["index"] = self.index
        _opt(d, "fullyQualifiedName", self.fully_qualified_name)
        _opt(d, "decoratedName", self.decorated_name)
        if self.parent_index >= 0:
            d["parentIndex"] = self.parent_index
        _opt(d, "kind", self.kind)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> LogicalLocation:
        return cls(
            name=str(d.get("name", "")),
            index=int(d.get("index", -1)),
            fully_qualified_name=str(d.get("fullyQualifiedName", "")),
            decorated_name=str(d.get("decoratedName", "")),
            parent_index=int(d.get("parentIndex", -1)),
            kind=str(d.get("kind", "")),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class PhysicalLocation:
    """Physical location within an artifact (§3.29)."""
    artifact_location: Optional[ArtifactLocation] = None
    region: Optional[Region] = None
    context_region: Optional[Region] = None
    address: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.artifact_location is not None:
            d["artifactLocation"] = self.artifact_location.to_dict()
        if self.region is not None:
            d["region"] = self.region.to_dict()
        if self.context_region is not None:
            d["contextRegion"] = self.context_region.to_dict()
        if self.address is not None:
            d["address"] = self.address
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> PhysicalLocation:
        al = d.get("artifactLocation")
        r = d.get("region")
        cr = d.get("contextRegion")
        return cls(
            artifact_location=ArtifactLocation.from_dict(al) if al else None,
            region=Region.from_dict(r) if r else None,
            context_region=Region.from_dict(cr) if cr else None,
            address=d.get("address"),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Location:
    """A SARIF location (§3.28)."""
    id: int = -1
    physical_location: Optional[PhysicalLocation] = None
    logical_locations: Tuple[LogicalLocation, ...] = ()
    message: Optional[Message] = None
    annotations: Tuple[Region, ...] = ()
    relationships: Tuple[Dict[str, Any], ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.id >= 0:
            d["id"] = self.id
        if self.physical_location is not None:
            d["physicalLocation"] = self.physical_location.to_dict()
        if self.logical_locations:
            d["logicalLocations"] = [ll.to_dict() for ll in self.logical_locations]
        if self.message is not None:
            d["message"] = self.message.to_dict()
        if self.annotations:
            d["annotations"] = [a.to_dict() for a in self.annotations]
        if self.relationships:
            d["relationships"] = list(self.relationships)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Location:
        pl = d.get("physicalLocation")
        msg = d.get("message")
        return cls(
            id=int(d.get("id", -1)),
            physical_location=PhysicalLocation.from_dict(pl) if pl else None,
            logical_locations=tuple(
                LogicalLocation.from_dict(ll)
                for ll in d.get("logicalLocations", [])
            ),
            message=Message.from_dict(msg) if msg else None,
            annotations=tuple(
                Region.from_dict(a) for a in d.get("annotations", [])
            ),
            relationships=tuple(d.get("relationships", [])),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Artifact
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Artifact:
    """A SARIF artifact (§3.24)."""
    location: Optional[ArtifactLocation] = None
    description: Optional[Message] = None
    mime_type: str = ""
    length: int = -1
    roles: Tuple[str, ...] = ()
    contents: Optional[ArtifactContent] = None
    encoding: str = ""
    source_language: str = ""
    hashes: Dict[str, str] = field(default_factory=dict)
    last_modified_time_utc: str = ""
    parent_index: int = -1
    offset: int = -1
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.location is not None:
            d["location"] = self.location.to_dict()
        if self.description is not None and self.description.text:
            d["description"] = self.description.to_dict()
        _opt(d, "mimeType", self.mime_type)
        if self.length >= 0:
            d["length"] = self.length
        if self.roles:
            d["roles"] = list(self.roles)
        if self.contents is not None:
            d["contents"] = self.contents.to_dict()
        _opt(d, "encoding", self.encoding)
        _opt(d, "sourceLanguage", self.source_language)
        if self.hashes:
            d["hashes"] = dict(self.hashes)
        _opt(d, "lastModifiedTimeUtc", self.last_modified_time_utc)
        if self.parent_index >= 0:
            d["parentIndex"] = self.parent_index
        if self.offset >= 0:
            d["offset"] = self.offset
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Artifact:
        loc = d.get("location")
        desc = d.get("description")
        cont = d.get("contents")
        return cls(
            location=ArtifactLocation.from_dict(loc) if loc else None,
            description=Message.from_dict(desc) if desc else None,
            mime_type=str(d.get("mimeType", "")),
            length=int(d.get("length", -1)),
            roles=tuple(d.get("roles", [])),
            contents=ArtifactContent.from_dict(cont) if cont else None,
            encoding=str(d.get("encoding", "")),
            source_language=str(d.get("sourceLanguage", "")),
            hashes=d.get("hashes", {}),
            last_modified_time_utc=str(d.get("lastModifiedTimeUtc", "")),
            parent_index=int(d.get("parentIndex", -1)),
            offset=int(d.get("offset", -1)),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Reporting descriptors (rules / notifications)
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ReportingConfiguration:
    """Default configuration for a reporting descriptor (§3.51)."""
    enabled: bool = True
    level: Level = Level.WARNING
    rank: float = -1.0
    parameters: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"enabled": self.enabled, "level": self.level.value}
        if self.rank >= 0:
            d["rank"] = self.rank
        _opt(d, "parameters", self.parameters)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReportingConfiguration:
        return cls(
            enabled=bool(d.get("enabled", True)),
            level=Level(d.get("level", "warning")),
            rank=float(d.get("rank", -1.0)),
            parameters=d.get("parameters", {}),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ReportingDescriptor:
    """A rule or notification descriptor (§3.49)."""
    id: str = ""
    name: str = ""
    short_description: Optional[MultiformatMessageString] = None
    full_description: Optional[MultiformatMessageString] = None
    message_strings: Dict[str, MultiformatMessageString] = field(
        default_factory=dict
    )
    default_configuration: Optional[ReportingConfiguration] = None
    help_uri: str = ""
    help: Optional[MultiformatMessageString] = None
    relationships: Tuple[Dict[str, Any], ...] = ()
    deprecated_ids: Tuple[str, ...] = ()
    deprecated_names: Tuple[str, ...] = ()
    guid: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id}
        _opt(d, "name", self.name)
        if self.short_description is not None:
            d["shortDescription"] = self.short_description.to_dict()
        if self.full_description is not None:
            d["fullDescription"] = self.full_description.to_dict()
        if self.message_strings:
            d["messageStrings"] = {
                k: v.to_dict() for k, v in self.message_strings.items()
            }
        if self.default_configuration is not None:
            d["defaultConfiguration"] = self.default_configuration.to_dict()
        _opt(d, "helpUri", self.help_uri)
        if self.help is not None:
            d["help"] = self.help.to_dict()
        if self.relationships:
            d["relationships"] = list(self.relationships)
        if self.deprecated_ids:
            d["deprecatedIds"] = list(self.deprecated_ids)
        if self.deprecated_names:
            d["deprecatedNames"] = list(self.deprecated_names)
        _opt(d, "guid", self.guid)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ReportingDescriptor:
        sd = d.get("shortDescription")
        fd = d.get("fullDescription")
        dc = d.get("defaultConfiguration")
        hlp = d.get("help")
        ms = d.get("messageStrings", {})
        return cls(
            id=str(d.get("id", "")),
            name=str(d.get("name", "")),
            short_description=(
                MultiformatMessageString.from_dict(sd) if sd else None
            ),
            full_description=(
                MultiformatMessageString.from_dict(fd) if fd else None
            ),
            message_strings={
                k: MultiformatMessageString.from_dict(v) for k, v in ms.items()
            },
            default_configuration=(
                ReportingConfiguration.from_dict(dc) if dc else None
            ),
            help_uri=str(d.get("helpUri", "")),
            help=MultiformatMessageString.from_dict(hlp) if hlp else None,
            relationships=tuple(d.get("relationships", [])),
            deprecated_ids=tuple(d.get("deprecatedIds", [])),
            deprecated_names=tuple(d.get("deprecatedNames", [])),
            guid=str(d.get("guid", "")),
            properties=d.get("properties", {}),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.id:
            errors.append("ReportingDescriptor.id is required")
        return errors


# ═══════════════════════════════════════════════════════════════════════════
# Tool
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ToolComponentReference:
    """Reference to a tool component (§3.54)."""
    name: str = ""
    index: int = -1
    guid: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "name", self.name)
        if self.index >= 0:
            d["index"] = self.index
        _opt(d, "guid", self.guid)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ToolComponentReference:
        return cls(
            name=str(d.get("name", "")),
            index=int(d.get("index", -1)),
            guid=str(d.get("guid", "")),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ToolComponent:
    """A SARIF tool component — driver or extension (§3.19)."""
    name: str = ""
    version: str = ""
    semantic_version: str = ""
    full_name: str = ""
    information_uri: str = ""
    download_uri: str = ""
    organization: str = ""
    product: str = ""
    product_suite: str = ""
    guid: str = ""
    short_description: Optional[MultiformatMessageString] = None
    full_description: Optional[MultiformatMessageString] = None
    rules: Tuple[ReportingDescriptor, ...] = ()
    notifications: Tuple[ReportingDescriptor, ...] = ()
    taxa: Tuple[ReportingDescriptor, ...] = ()
    supported_taxonomies: Tuple[ToolComponentReference, ...] = ()
    language: str = "en-US"
    contents: Tuple[str, ...] = ("localizedData", "nonLocalizedData")
    is_comprehensive: bool = False
    released_date_utc: str = ""
    associated_component: Optional[ToolComponentReference] = None
    translation_metadata: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"name": self.name}
        _opt(d, "version", self.version)
        _opt(d, "semanticVersion", self.semantic_version)
        _opt(d, "fullName", self.full_name)
        _opt(d, "informationUri", self.information_uri)
        _opt(d, "downloadUri", self.download_uri)
        _opt(d, "organization", self.organization)
        _opt(d, "product", self.product)
        _opt(d, "productSuite", self.product_suite)
        _opt(d, "guid", self.guid)
        if self.short_description is not None:
            d["shortDescription"] = self.short_description.to_dict()
        if self.full_description is not None:
            d["fullDescription"] = self.full_description.to_dict()
        if self.rules:
            d["rules"] = [r.to_dict() for r in self.rules]
        if self.notifications:
            d["notifications"] = [n.to_dict() for n in self.notifications]
        if self.taxa:
            d["taxa"] = [t.to_dict() for t in self.taxa]
        if self.supported_taxonomies:
            d["supportedTaxonomies"] = [
                t.to_dict() for t in self.supported_taxonomies
            ]
        _opt(d, "language", self.language if self.language != "en-US" else "")
        if self.is_comprehensive:
            d["isComprehensive"] = True
        _opt(d, "releasedDateUtc", self.released_date_utc)
        if self.associated_component is not None:
            d["associatedComponent"] = self.associated_component.to_dict()
        if self.translation_metadata is not None:
            d["translationMetadata"] = self.translation_metadata
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ToolComponent:
        sd = d.get("shortDescription")
        fd = d.get("fullDescription")
        ac = d.get("associatedComponent")
        return cls(
            name=str(d.get("name", "")),
            version=str(d.get("version", "")),
            semantic_version=str(d.get("semanticVersion", "")),
            full_name=str(d.get("fullName", "")),
            information_uri=str(d.get("informationUri", "")),
            download_uri=str(d.get("downloadUri", "")),
            organization=str(d.get("organization", "")),
            product=str(d.get("product", "")),
            product_suite=str(d.get("productSuite", "")),
            guid=str(d.get("guid", "")),
            short_description=(
                MultiformatMessageString.from_dict(sd) if sd else None
            ),
            full_description=(
                MultiformatMessageString.from_dict(fd) if fd else None
            ),
            rules=tuple(
                ReportingDescriptor.from_dict(r) for r in d.get("rules", [])
            ),
            notifications=tuple(
                ReportingDescriptor.from_dict(n)
                for n in d.get("notifications", [])
            ),
            taxa=tuple(
                ReportingDescriptor.from_dict(t) for t in d.get("taxa", [])
            ),
            supported_taxonomies=tuple(
                ToolComponentReference.from_dict(t)
                for t in d.get("supportedTaxonomies", [])
            ),
            language=str(d.get("language", "en-US")),
            contents=tuple(d.get("contents", ("localizedData", "nonLocalizedData"))),
            is_comprehensive=bool(d.get("isComprehensive", False)),
            released_date_utc=str(d.get("releasedDateUtc", "")),
            associated_component=(
                ToolComponentReference.from_dict(ac) if ac else None
            ),
            translation_metadata=d.get("translationMetadata"),
            properties=d.get("properties", {}),
        )

    def validate(self) -> List[str]:
        errors: List[str] = []
        if not self.name:
            errors.append("ToolComponent.name is required")
        for i, r in enumerate(self.rules):
            errors.extend(
                f"rules[{i}].{e}" for e in r.validate()
            )
        return errors


@dataclass(frozen=True, slots=True)
class Tool:
    """The analysis tool that produced the run (§3.18)."""
    driver: ToolComponent = field(default_factory=ToolComponent)
    extensions: Tuple[ToolComponent, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"driver": self.driver.to_dict()}
        if self.extensions:
            d["extensions"] = [e.to_dict() for e in self.extensions]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Tool:
        drv = d.get("driver", {})
        return cls(
            driver=ToolComponent.from_dict(drv),
            extensions=tuple(
                ToolComponent.from_dict(e) for e in d.get("extensions", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Fix / Replacement
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Replacement:
    """A replacement within an artifact (§3.57)."""
    deleted_region: Region = field(default_factory=Region)
    inserted_content: Optional[ArtifactContent] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"deletedRegion": self.deleted_region.to_dict()}
        if self.inserted_content is not None:
            d["insertedContent"] = self.inserted_content.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Replacement:
        dr = d.get("deletedRegion", {})
        ic = d.get("insertedContent")
        return cls(
            deleted_region=Region.from_dict(dr),
            inserted_content=ArtifactContent.from_dict(ic) if ic else None,
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ArtifactChange:
    """Changes to a single artifact (§3.56)."""
    artifact_location: ArtifactLocation = field(default_factory=ArtifactLocation)
    replacements: Tuple[Replacement, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "artifactLocation": self.artifact_location.to_dict(),
            "replacements": [r.to_dict() for r in self.replacements],
        }
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ArtifactChange:
        al = d.get("artifactLocation", {})
        return cls(
            artifact_location=ArtifactLocation.from_dict(al),
            replacements=tuple(
                Replacement.from_dict(r) for r in d.get("replacements", [])
            ),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Fix:
    """A proposed fix for a result (§3.55)."""
    description: Message = field(default_factory=Message)
    artifact_changes: Tuple[ArtifactChange, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"description": self.description.to_dict()}
        if self.artifact_changes:
            d["artifactChanges"] = [c.to_dict() for c in self.artifact_changes]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Fix:
        desc = d.get("description", {})
        return cls(
            description=Message.from_dict(desc),
            artifact_changes=tuple(
                ArtifactChange.from_dict(c)
                for c in d.get("artifactChanges", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Code flow / thread flow
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class StackFrame:
    """A single frame in a call stack (§3.47)."""
    location: Optional[Location] = None
    module: str = ""
    thread_id: int = -1
    parameters: Tuple[str, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.location is not None:
            d["location"] = self.location.to_dict()
        _opt(d, "module", self.module)
        if self.thread_id >= 0:
            d["threadId"] = self.thread_id
        if self.parameters:
            d["parameters"] = list(self.parameters)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> StackFrame:
        loc = d.get("location")
        return cls(
            location=Location.from_dict(loc) if loc else None,
            module=str(d.get("module", "")),
            thread_id=int(d.get("threadId", -1)),
            parameters=tuple(d.get("parameters", [])),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Stack:
    """A call stack (§3.44)."""
    message: Optional[Message] = None
    frames: Tuple[StackFrame, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"frames": [f.to_dict() for f in self.frames]}
        if self.message is not None:
            d["message"] = self.message.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Stack:
        msg = d.get("message")
        return cls(
            message=Message.from_dict(msg) if msg else None,
            frames=tuple(StackFrame.from_dict(f) for f in d.get("frames", [])),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ThreadFlowLocation:
    """A location within a thread flow (§3.38)."""
    index: int = -1
    location: Optional[Location] = None
    stack: Optional[Stack] = None
    kinds: Tuple[str, ...] = ()
    taxa: Tuple[Dict[str, Any], ...] = ()
    module: str = ""
    state: Dict[str, MultiformatMessageString] = field(default_factory=dict)
    nesting_level: int = 0
    execution_order: int = -1
    execution_time_utc: str = ""
    importance: Optional[ThreadFlowImportance] = None
    web_request: Optional[Dict[str, Any]] = None
    web_response: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.index >= 0:
            d["index"] = self.index
        if self.location is not None:
            d["location"] = self.location.to_dict()
        if self.stack is not None:
            d["stack"] = self.stack.to_dict()
        if self.kinds:
            d["kinds"] = list(self.kinds)
        if self.taxa:
            d["taxa"] = list(self.taxa)
        _opt(d, "module", self.module)
        if self.state:
            d["state"] = {k: v.to_dict() for k, v in self.state.items()}
        if self.nesting_level > 0:
            d["nestingLevel"] = self.nesting_level
        if self.execution_order >= 0:
            d["executionOrder"] = self.execution_order
        _opt(d, "executionTimeUtc", self.execution_time_utc)
        if self.importance is not None:
            d["importance"] = self.importance.value
        if self.web_request is not None:
            d["webRequest"] = self.web_request
        if self.web_response is not None:
            d["webResponse"] = self.web_response
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ThreadFlowLocation:
        loc = d.get("location")
        stk = d.get("stack")
        imp = d.get("importance")
        st = d.get("state", {})
        return cls(
            index=int(d.get("index", -1)),
            location=Location.from_dict(loc) if loc else None,
            stack=Stack.from_dict(stk) if stk else None,
            kinds=tuple(d.get("kinds", [])),
            taxa=tuple(d.get("taxa", [])),
            module=str(d.get("module", "")),
            state={
                k: MultiformatMessageString.from_dict(v)
                for k, v in st.items()
            },
            nesting_level=int(d.get("nestingLevel", 0)),
            execution_order=int(d.get("executionOrder", -1)),
            execution_time_utc=str(d.get("executionTimeUtc", "")),
            importance=ThreadFlowImportance(imp) if imp else None,
            web_request=d.get("webRequest"),
            web_response=d.get("webResponse"),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ThreadFlow:
    """A sequence of code locations forming a flow (§3.37)."""
    id: str = ""
    message: Optional[Message] = None
    locations: Tuple[ThreadFlowLocation, ...] = ()
    initial_state: Dict[str, MultiformatMessageString] = field(
        default_factory=dict
    )
    immutable_state: Dict[str, MultiformatMessageString] = field(
        default_factory=dict
    )
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "locations": [loc.to_dict() for loc in self.locations],
        }
        _opt(d, "id", self.id)
        if self.message is not None:
            d["message"] = self.message.to_dict()
        if self.initial_state:
            d["initialState"] = {
                k: v.to_dict() for k, v in self.initial_state.items()
            }
        if self.immutable_state:
            d["immutableState"] = {
                k: v.to_dict() for k, v in self.immutable_state.items()
            }
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ThreadFlow:
        msg = d.get("message")
        init_s = d.get("initialState", {})
        imm_s = d.get("immutableState", {})
        return cls(
            id=str(d.get("id", "")),
            message=Message.from_dict(msg) if msg else None,
            locations=tuple(
                ThreadFlowLocation.from_dict(loc)
                for loc in d.get("locations", [])
            ),
            initial_state={
                k: MultiformatMessageString.from_dict(v)
                for k, v in init_s.items()
            },
            immutable_state={
                k: MultiformatMessageString.from_dict(v)
                for k, v in imm_s.items()
            },
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class CodeFlow:
    """A set of thread flows describing a code path (§3.36)."""
    message: Optional[Message] = None
    thread_flows: Tuple[ThreadFlow, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "threadFlows": [tf.to_dict() for tf in self.thread_flows],
        }
        if self.message is not None:
            d["message"] = self.message.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> CodeFlow:
        msg = d.get("message")
        return cls(
            message=Message.from_dict(msg) if msg else None,
            thread_flows=tuple(
                ThreadFlow.from_dict(tf) for tf in d.get("threadFlows", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Graph types
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Node:
    """A node in a graph (§3.41)."""
    id: str = ""
    label: Optional[Message] = None
    location: Optional[Location] = None
    children: Tuple[Node, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"id": self.id}
        if self.label is not None:
            d["label"] = self.label.to_dict()
        if self.location is not None:
            d["location"] = self.location.to_dict()
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Node:
        lbl = d.get("label")
        loc = d.get("location")
        return cls(
            id=str(d.get("id", "")),
            label=Message.from_dict(lbl) if lbl else None,
            location=Location.from_dict(loc) if loc else None,
            children=tuple(Node.from_dict(c) for c in d.get("children", [])),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Edge:
    """An edge in a graph (§3.42)."""
    id: str = ""
    label: Optional[Message] = None
    source_node_id: str = ""
    target_node_id: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "id": self.id,
            "sourceNodeId": self.source_node_id,
            "targetNodeId": self.target_node_id,
        }
        if self.label is not None:
            d["label"] = self.label.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Edge:
        lbl = d.get("label")
        return cls(
            id=str(d.get("id", "")),
            label=Message.from_dict(lbl) if lbl else None,
            source_node_id=str(d.get("sourceNodeId", "")),
            target_node_id=str(d.get("targetNodeId", "")),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Graph:
    """A graph associated with a result or run (§3.39)."""
    description: Optional[Message] = None
    nodes: Tuple[Node, ...] = ()
    edges: Tuple[Edge, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        if self.description is not None:
            d["description"] = self.description.to_dict()
        if self.nodes:
            d["nodes"] = [n.to_dict() for n in self.nodes]
        if self.edges:
            d["edges"] = [e.to_dict() for e in self.edges]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Graph:
        desc = d.get("description")
        return cls(
            description=Message.from_dict(desc) if desc else None,
            nodes=tuple(Node.from_dict(n) for n in d.get("nodes", [])),
            edges=tuple(Edge.from_dict(e) for e in d.get("edges", [])),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Notification / ExceptionData
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class ExceptionData:
    """Runtime exception data (§3.58.2)."""
    kind: str = ""
    message: str = ""
    stack: Optional[Stack] = None
    inner_exceptions: Tuple[ExceptionData, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "kind", self.kind)
        _opt(d, "message", self.message)
        if self.stack is not None:
            d["stack"] = self.stack.to_dict()
        if self.inner_exceptions:
            d["innerExceptions"] = [e.to_dict() for e in self.inner_exceptions]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ExceptionData:
        stk = d.get("stack")
        return cls(
            kind=str(d.get("kind", "")),
            message=str(d.get("message", "")),
            stack=Stack.from_dict(stk) if stk else None,
            inner_exceptions=tuple(
                ExceptionData.from_dict(e)
                for e in d.get("innerExceptions", [])
            ),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class Notification:
    """A notification emitted by the tool (§3.58)."""
    descriptor: Optional[Dict[str, Any]] = None
    associated_rule: Optional[Dict[str, Any]] = None
    locations: Tuple[Location, ...] = ()
    message: Message = field(default_factory=Message)
    level: NotificationLevel = NotificationLevel.WARNING
    thread_id: int = -1
    time_utc: str = ""
    exception_data: Optional[ExceptionData] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "message": self.message.to_dict(),
            "level": self.level.value,
        }
        if self.descriptor is not None:
            d["descriptor"] = self.descriptor
        if self.associated_rule is not None:
            d["associatedRule"] = self.associated_rule
        if self.locations:
            d["locations"] = [loc.to_dict() for loc in self.locations]
        if self.thread_id >= 0:
            d["threadId"] = self.thread_id
        _opt(d, "timeUtc", self.time_utc)
        if self.exception_data is not None:
            d["exception"] = self.exception_data.to_dict()
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Notification:
        msg = d.get("message", {})
        exc = d.get("exception")
        return cls(
            descriptor=d.get("descriptor"),
            associated_rule=d.get("associatedRule"),
            locations=tuple(
                Location.from_dict(loc) for loc in d.get("locations", [])
            ),
            message=Message.from_dict(msg),
            level=NotificationLevel(d.get("level", "warning")),
            thread_id=int(d.get("threadId", -1)),
            time_utc=str(d.get("timeUtc", "")),
            exception_data=ExceptionData.from_dict(exc) if exc else None,
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Invocation
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Invocation:
    """A SARIF invocation record (§3.20)."""
    command_line: str = ""
    arguments: Tuple[str, ...] = ()
    response_files: Tuple[ArtifactLocation, ...] = ()
    start_time_utc: str = ""
    end_time_utc: str = ""
    execution_successful: bool = True
    exit_code: int = 0
    exit_code_description: str = ""
    exit_signal_name: str = ""
    exit_signal_number: int = -1
    process_start_failure_message: str = ""
    machine: str = ""
    account: str = ""
    process_id: int = -1
    working_directory: Optional[ArtifactLocation] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    stdin: Optional[ArtifactLocation] = None
    stdout: Optional[ArtifactLocation] = None
    stderr: Optional[ArtifactLocation] = None
    stdout_stderr: Optional[ArtifactLocation] = None
    executable_location: Optional[ArtifactLocation] = None
    tool_execution_notifications: Tuple[Notification, ...] = ()
    tool_configuration_notifications: Tuple[Notification, ...] = ()
    notification_properties: Dict[str, Any] = field(default_factory=dict)
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "executionSuccessful": self.execution_successful,
        }
        _opt(d, "commandLine", self.command_line)
        if self.arguments:
            d["arguments"] = list(self.arguments)
        if self.response_files:
            d["responseFiles"] = [rf.to_dict() for rf in self.response_files]
        _opt(d, "startTimeUtc", self.start_time_utc)
        _opt(d, "endTimeUtc", self.end_time_utc)
        if self.exit_code != 0:
            d["exitCode"] = self.exit_code
        _opt(d, "exitCodeDescription", self.exit_code_description)
        _opt(d, "exitSignalName", self.exit_signal_name)
        if self.exit_signal_number >= 0:
            d["exitSignalNumber"] = self.exit_signal_number
        _opt(d, "processStartFailureMessage", self.process_start_failure_message)
        _opt(d, "machine", self.machine)
        _opt(d, "account", self.account)
        if self.process_id >= 0:
            d["processId"] = self.process_id
        if self.working_directory is not None:
            d["workingDirectory"] = self.working_directory.to_dict()
        if self.environment_variables:
            d["environmentVariables"] = dict(self.environment_variables)
        if self.stdin is not None:
            d["stdin"] = self.stdin.to_dict()
        if self.stdout is not None:
            d["stdout"] = self.stdout.to_dict()
        if self.stderr is not None:
            d["stderr"] = self.stderr.to_dict()
        if self.stdout_stderr is not None:
            d["stdoutStderr"] = self.stdout_stderr.to_dict()
        if self.executable_location is not None:
            d["executableLocation"] = self.executable_location.to_dict()
        if self.tool_execution_notifications:
            d["toolExecutionNotifications"] = [
                n.to_dict() for n in self.tool_execution_notifications
            ]
        if self.tool_configuration_notifications:
            d["toolConfigurationNotifications"] = [
                n.to_dict() for n in self.tool_configuration_notifications
            ]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Invocation:
        wd = d.get("workingDirectory")
        stdin_ = d.get("stdin")
        stdout_ = d.get("stdout")
        stderr_ = d.get("stderr")
        stdout_stderr_ = d.get("stdoutStderr")
        exe = d.get("executableLocation")
        return cls(
            command_line=str(d.get("commandLine", "")),
            arguments=tuple(d.get("arguments", [])),
            response_files=tuple(
                ArtifactLocation.from_dict(rf)
                for rf in d.get("responseFiles", [])
            ),
            start_time_utc=str(d.get("startTimeUtc", "")),
            end_time_utc=str(d.get("endTimeUtc", "")),
            execution_successful=bool(d.get("executionSuccessful", True)),
            exit_code=int(d.get("exitCode", 0)),
            exit_code_description=str(d.get("exitCodeDescription", "")),
            exit_signal_name=str(d.get("exitSignalName", "")),
            exit_signal_number=int(d.get("exitSignalNumber", -1)),
            process_start_failure_message=str(
                d.get("processStartFailureMessage", "")
            ),
            machine=str(d.get("machine", "")),
            account=str(d.get("account", "")),
            process_id=int(d.get("processId", -1)),
            working_directory=(
                ArtifactLocation.from_dict(wd) if wd else None
            ),
            environment_variables=d.get("environmentVariables", {}),
            stdin=ArtifactLocation.from_dict(stdin_) if stdin_ else None,
            stdout=ArtifactLocation.from_dict(stdout_) if stdout_ else None,
            stderr=ArtifactLocation.from_dict(stderr_) if stderr_ else None,
            stdout_stderr=(
                ArtifactLocation.from_dict(stdout_stderr_)
                if stdout_stderr_
                else None
            ),
            executable_location=(
                ArtifactLocation.from_dict(exe) if exe else None
            ),
            tool_execution_notifications=tuple(
                Notification.from_dict(n)
                for n in d.get("toolExecutionNotifications", [])
            ),
            tool_configuration_notifications=tuple(
                Notification.from_dict(n)
                for n in d.get("toolConfigurationNotifications", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Suppression / ResultProvenance
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Suppression:
    """A result suppression (§3.35)."""
    kind: SuppressionKind = SuppressionKind.IN_SOURCE
    status: Optional[SuppressionStatus] = None
    location: Optional[Location] = None
    guid: str = ""
    justification: str = ""
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"kind": self.kind.value}
        if self.status is not None:
            d["status"] = self.status.value
        if self.location is not None:
            d["location"] = self.location.to_dict()
        _opt(d, "guid", self.guid)
        _opt(d, "justification", self.justification)
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Suppression:
        loc = d.get("location")
        sts = d.get("status")
        return cls(
            kind=SuppressionKind(d.get("kind", "inSource")),
            status=SuppressionStatus(sts) if sts else None,
            location=Location.from_dict(loc) if loc else None,
            guid=str(d.get("guid", "")),
            justification=str(d.get("justification", "")),
            properties=d.get("properties", {}),
        )


@dataclass(frozen=True, slots=True)
class ResultProvenance:
    """Provenance information for a result (§3.48)."""
    first_detection_time_utc: str = ""
    last_detection_time_utc: str = ""
    first_detection_run_guid: str = ""
    last_detection_run_guid: str = ""
    invocation_index: int = -1
    conversion_sources: Tuple[PhysicalLocation, ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {}
        _opt(d, "firstDetectionTimeUtc", self.first_detection_time_utc)
        _opt(d, "lastDetectionTimeUtc", self.last_detection_time_utc)
        _opt(d, "firstDetectionRunGuid", self.first_detection_run_guid)
        _opt(d, "lastDetectionRunGuid", self.last_detection_run_guid)
        if self.invocation_index >= 0:
            d["invocationIndex"] = self.invocation_index
        if self.conversion_sources:
            d["conversionSources"] = [
                cs.to_dict() for cs in self.conversion_sources
            ]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> ResultProvenance:
        return cls(
            first_detection_time_utc=str(
                d.get("firstDetectionTimeUtc", "")
            ),
            last_detection_time_utc=str(d.get("lastDetectionTimeUtc", "")),
            first_detection_run_guid=str(
                d.get("firstDetectionRunGuid", "")
            ),
            last_detection_run_guid=str(d.get("lastDetectionRunGuid", "")),
            invocation_index=int(d.get("invocationIndex", -1)),
            conversion_sources=tuple(
                PhysicalLocation.from_dict(cs)
                for cs in d.get("conversionSources", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Result
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Result:
    """A single SARIF result (§3.27)."""
    rule_id: str = ""
    rule_index: int = -1
    rule: Optional[Dict[str, Any]] = None
    kind: Kind = Kind.FAIL
    level: Level = Level.WARNING
    message: Message = field(default_factory=Message)
    analysis_target: Optional[ArtifactLocation] = None
    locations: Tuple[Location, ...] = ()
    guid: str = ""
    correlation_guid: str = ""
    occurrence_count: int = -1
    partial_fingerprints: Dict[str, str] = field(default_factory=dict)
    fingerprints: Dict[str, str] = field(default_factory=dict)
    stacks: Tuple[Stack, ...] = ()
    code_flows: Tuple[CodeFlow, ...] = ()
    graphs: Tuple[Graph, ...] = ()
    graph_traversals: Tuple[Dict[str, Any], ...] = ()
    related_locations: Tuple[Location, ...] = ()
    suppressions: Tuple[Suppression, ...] = ()
    baseline_state: Optional[BaselineState] = None
    rank: float = -1.0
    attachments: Tuple[Dict[str, Any], ...] = ()
    hosted_viewer_uri: str = ""
    work_item_uris: Tuple[str, ...] = ()
    provenance: Optional[ResultProvenance] = None
    fixes: Tuple[Fix, ...] = ()
    taxa: Tuple[Dict[str, Any], ...] = ()
    web_request: Optional[Dict[str, Any]] = None
    web_response: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "message": self.message.to_dict(),
        }
        _opt(d, "ruleId", self.rule_id)
        if self.rule_index >= 0:
            d["ruleIndex"] = self.rule_index
        if self.rule is not None:
            d["rule"] = self.rule
        d["kind"] = self.kind.value
        d["level"] = self.level.value
        if self.analysis_target is not None:
            d["analysisTarget"] = self.analysis_target.to_dict()
        if self.locations:
            d["locations"] = [loc.to_dict() for loc in self.locations]
        _opt(d, "guid", self.guid)
        _opt(d, "correlationGuid", self.correlation_guid)
        if self.occurrence_count >= 0:
            d["occurrenceCount"] = self.occurrence_count
        if self.partial_fingerprints:
            d["partialFingerprints"] = dict(self.partial_fingerprints)
        if self.fingerprints:
            d["fingerprints"] = dict(self.fingerprints)
        if self.stacks:
            d["stacks"] = [s.to_dict() for s in self.stacks]
        if self.code_flows:
            d["codeFlows"] = [cf.to_dict() for cf in self.code_flows]
        if self.graphs:
            d["graphs"] = [g.to_dict() for g in self.graphs]
        if self.graph_traversals:
            d["graphTraversals"] = list(self.graph_traversals)
        if self.related_locations:
            d["relatedLocations"] = [
                rl.to_dict() for rl in self.related_locations
            ]
        if self.suppressions:
            d["suppressions"] = [s.to_dict() for s in self.suppressions]
        if self.baseline_state is not None:
            d["baselineState"] = self.baseline_state.value
        if self.rank >= 0:
            d["rank"] = self.rank
        if self.attachments:
            d["attachments"] = list(self.attachments)
        _opt(d, "hostedViewerUri", self.hosted_viewer_uri)
        if self.work_item_uris:
            d["workItemUris"] = list(self.work_item_uris)
        if self.provenance is not None:
            d["provenance"] = self.provenance.to_dict()
        if self.fixes:
            d["fixes"] = [f.to_dict() for f in self.fixes]
        if self.taxa:
            d["taxa"] = list(self.taxa)
        if self.web_request is not None:
            d["webRequest"] = self.web_request
        if self.web_response is not None:
            d["webResponse"] = self.web_response
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Result:
        msg = d.get("message", {})
        at = d.get("analysisTarget")
        bs = d.get("baselineState")
        prov = d.get("provenance")
        return cls(
            rule_id=str(d.get("ruleId", "")),
            rule_index=int(d.get("ruleIndex", -1)),
            rule=d.get("rule"),
            kind=Kind(d.get("kind", "fail")),
            level=Level(d.get("level", "warning")),
            message=Message.from_dict(msg),
            analysis_target=(
                ArtifactLocation.from_dict(at) if at else None
            ),
            locations=tuple(
                Location.from_dict(loc) for loc in d.get("locations", [])
            ),
            guid=str(d.get("guid", "")),
            correlation_guid=str(d.get("correlationGuid", "")),
            occurrence_count=int(d.get("occurrenceCount", -1)),
            partial_fingerprints=d.get("partialFingerprints", {}),
            fingerprints=d.get("fingerprints", {}),
            stacks=tuple(Stack.from_dict(s) for s in d.get("stacks", [])),
            code_flows=tuple(
                CodeFlow.from_dict(cf) for cf in d.get("codeFlows", [])
            ),
            graphs=tuple(
                Graph.from_dict(g) for g in d.get("graphs", [])
            ),
            graph_traversals=tuple(d.get("graphTraversals", [])),
            related_locations=tuple(
                Location.from_dict(rl) for rl in d.get("relatedLocations", [])
            ),
            suppressions=tuple(
                Suppression.from_dict(s) for s in d.get("suppressions", [])
            ),
            baseline_state=BaselineState(bs) if bs else None,
            rank=float(d.get("rank", -1.0)),
            attachments=tuple(d.get("attachments", [])),
            hosted_viewer_uri=str(d.get("hostedViewerUri", "")),
            work_item_uris=tuple(d.get("workItemUris", [])),
            provenance=(
                ResultProvenance.from_dict(prov) if prov else None
            ),
            fixes=tuple(Fix.from_dict(f) for f in d.get("fixes", [])),
            taxa=tuple(d.get("taxa", [])),
            web_request=d.get("webRequest"),
            web_response=d.get("webResponse"),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# Taxonomy
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Taxonomy:
    """An external taxonomy referenced by the run (modelled as ToolComponent)."""
    component: ToolComponent = field(default_factory=ToolComponent)

    def to_dict(self) -> Dict[str, Any]:
        return self.component.to_dict()

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Taxonomy:
        return cls(component=ToolComponent.from_dict(d))


# ═══════════════════════════════════════════════════════════════════════════
# Run
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class Run:
    """A single analysis run (§3.14)."""
    tool: Tool = field(default_factory=Tool)
    invocations: Tuple[Invocation, ...] = ()
    conversion: Optional[Dict[str, Any]] = None
    language: str = "en-US"
    version_control_provenance: Tuple[Dict[str, Any], ...] = ()
    original_uri_base_ids: Dict[str, ArtifactLocation] = field(
        default_factory=dict
    )
    artifacts: Tuple[Artifact, ...] = ()
    logical_locations: Tuple[LogicalLocation, ...] = ()
    graphs: Tuple[Graph, ...] = ()
    results: Tuple[Result, ...] = ()
    automation_details: Optional[Dict[str, Any]] = None
    run_aggregates: Tuple[Dict[str, Any], ...] = ()
    baseline_guid: str = ""
    redaction_tokens: Tuple[str, ...] = ()
    default_encoding: str = ""
    default_source_language: str = ""
    newline_sequences: Tuple[str, ...] = ("\r\n", "\n")
    column_kind: str = "utf16CodeUnits"
    external_property_file_references: Optional[Dict[str, Any]] = None
    thread_flow_locations: Tuple[ThreadFlowLocation, ...] = ()
    taxonomies: Tuple[ToolComponent, ...] = ()
    addresses: Tuple[Dict[str, Any], ...] = ()
    translations: Tuple[ToolComponent, ...] = ()
    policies: Tuple[ToolComponent, ...] = ()
    web_requests: Tuple[Dict[str, Any], ...] = ()
    web_responses: Tuple[Dict[str, Any], ...] = ()
    special_locations: Optional[Dict[str, Any]] = None
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def num_results(self) -> int:
        return len(self.results)

    @property
    def num_errors(self) -> int:
        return sum(1 for r in self.results if r.level == Level.ERROR)

    @property
    def num_warnings(self) -> int:
        return sum(1 for r in self.results if r.level == Level.WARNING)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"tool": self.tool.to_dict()}
        if self.invocations:
            d["invocations"] = [inv.to_dict() for inv in self.invocations]
        if self.conversion is not None:
            d["conversion"] = self.conversion
        if self.language != "en-US":
            d["language"] = self.language
        if self.version_control_provenance:
            d["versionControlProvenance"] = list(
                self.version_control_provenance
            )
        if self.original_uri_base_ids:
            d["originalUriBaseIds"] = {
                k: v.to_dict() for k, v in self.original_uri_base_ids.items()
            }
        if self.artifacts:
            d["artifacts"] = [a.to_dict() for a in self.artifacts]
        if self.logical_locations:
            d["logicalLocations"] = [
                ll.to_dict() for ll in self.logical_locations
            ]
        if self.graphs:
            d["graphs"] = [g.to_dict() for g in self.graphs]
        # Always include results (may be empty array).
        d["results"] = [r.to_dict() for r in self.results]
        if self.automation_details is not None:
            d["automationDetails"] = self.automation_details
        if self.run_aggregates:
            d["runAggregates"] = list(self.run_aggregates)
        _opt(d, "baselineGuid", self.baseline_guid)
        if self.redaction_tokens:
            d["redactionTokens"] = list(self.redaction_tokens)
        _opt(d, "defaultEncoding", self.default_encoding)
        _opt(d, "defaultSourceLanguage", self.default_source_language)
        if self.column_kind != "utf16CodeUnits":
            d["columnKind"] = self.column_kind
        if self.external_property_file_references is not None:
            d["externalPropertyFileReferences"] = (
                self.external_property_file_references
            )
        if self.taxonomies:
            d["taxonomies"] = [t.to_dict() for t in self.taxonomies]
        if self.policies:
            d["policies"] = [p.to_dict() for p in self.policies]
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> Run:
        tool_d = d.get("tool", {})
        oub = d.get("originalUriBaseIds", {})
        return cls(
            tool=Tool.from_dict(tool_d),
            invocations=tuple(
                Invocation.from_dict(inv)
                for inv in d.get("invocations", [])
            ),
            conversion=d.get("conversion"),
            language=str(d.get("language", "en-US")),
            version_control_provenance=tuple(
                d.get("versionControlProvenance", [])
            ),
            original_uri_base_ids={
                k: ArtifactLocation.from_dict(v) for k, v in oub.items()
            },
            artifacts=tuple(
                Artifact.from_dict(a) for a in d.get("artifacts", [])
            ),
            logical_locations=tuple(
                LogicalLocation.from_dict(ll)
                for ll in d.get("logicalLocations", [])
            ),
            graphs=tuple(
                Graph.from_dict(g) for g in d.get("graphs", [])
            ),
            results=tuple(
                Result.from_dict(r) for r in d.get("results", [])
            ),
            automation_details=d.get("automationDetails"),
            run_aggregates=tuple(d.get("runAggregates", [])),
            baseline_guid=str(d.get("baselineGuid", "")),
            redaction_tokens=tuple(d.get("redactionTokens", [])),
            default_encoding=str(d.get("defaultEncoding", "")),
            default_source_language=str(d.get("defaultSourceLanguage", "")),
            newline_sequences=tuple(
                d.get("newlineSequences", ("\r\n", "\n"))
            ),
            column_kind=str(d.get("columnKind", "utf16CodeUnits")),
            external_property_file_references=d.get(
                "externalPropertyFileReferences"
            ),
            taxonomies=tuple(
                ToolComponent.from_dict(t) for t in d.get("taxonomies", [])
            ),
            policies=tuple(
                ToolComponent.from_dict(p) for p in d.get("policies", [])
            ),
            properties=d.get("properties", {}),
        )


# ═══════════════════════════════════════════════════════════════════════════
# SarifLog — top-level document
# ═══════════════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class SarifLog:
    """Top-level SARIF 2.1.0 log object (§3.13)."""
    version: str = SARIF_VERSION
    schema_uri: str = SARIF_SCHEMA_URI
    runs: Tuple[Run, ...] = ()
    inline_external_properties: Tuple[Dict[str, Any], ...] = ()
    properties: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_results(self) -> int:
        return sum(run.num_results for run in self.runs)

    @property
    def total_errors(self) -> int:
        return sum(run.num_errors for run in self.runs)

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "$schema": self.schema_uri,
            "version": self.version,
            "runs": [run.to_dict() for run in self.runs],
        }
        if self.inline_external_properties:
            d["inlineExternalProperties"] = list(
                self.inline_external_properties
            )
        _opt(d, "properties", self.properties)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifLog:
        return cls(
            version=str(d.get("version", SARIF_VERSION)),
            schema_uri=str(d.get("$schema", SARIF_SCHEMA_URI)),
            runs=tuple(Run.from_dict(r) for r in d.get("runs", [])),
            inline_external_properties=tuple(
                d.get("inlineExternalProperties", [])
            ),
            properties=d.get("properties", {}),
        )

    def validate(self) -> List[str]:
        """Return a list of validation errors."""
        errors: List[str] = []
        if self.version not in ("2.1.0", "2.0.0"):
            errors.append(f"Unsupported SARIF version: {self.version}")
        if not self.runs:
            errors.append("SARIF log must contain at least one run")
        for i, run in enumerate(self.runs):
            errs = run.tool.driver.validate()
            errors.extend(f"runs[{i}].tool.driver.{e}" for e in errs)
        return errors
