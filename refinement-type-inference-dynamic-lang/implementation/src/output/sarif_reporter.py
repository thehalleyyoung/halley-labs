from __future__ import annotations

import abc
import base64
import collections
import copy
import csv
import datetime
import enum
import hashlib
import io
import json
import os
import re
import textwrap
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
    Union,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SARIF_SCHEMA = "https://docs.oasis-open.org/sarif/sarif/v2.1.0/errata01/os/schemas/sarif-schema-2.1.0.json"
SARIF_VERSION = "2.1.0"
TOOL_NAME = "refinement-type-analyzer"
TOOL_VERSION = "0.5.0"
TOOL_SEMANTIC_VERSION = "0.5.0"
TOOL_INFO_URI = "https://github.com/refinement-types/analyzer"
TOOL_ORG = "Refinement Types Project"

MAX_GITHUB_ANNOTATIONS_PER_CALL = 50

# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class SarifLevel(enum.Enum):
    ERROR = "error"
    WARNING = "warning"
    NOTE = "note"
    NONE = "none"


class ArtifactRole(enum.Enum):
    ANALYSIS_TARGET = "analysisTarget"
    RESULT_FILE = "resultFile"
    RESPONSE_FILE = "responseFile"
    DEBUG_OUTPUT_FILE = "debugOutputFile"
    LOG_OUTPUT_FILE = "logOutputFile"
    TRACED_FILE = "tracedFile"
    POLICY = "policy"
    REFERENCED_ON_COMMAND_LINE = "referencedOnCommandLine"


class LogicalLocationKind(enum.Enum):
    FUNCTION = "function"
    MEMBER = "member"
    MODULE = "module"
    NAMESPACE = "namespace"
    TYPE = "type"
    RETURN_TYPE = "returnType"
    PARAMETER = "parameter"
    VARIABLE = "variable"


class SuppressionKind(enum.Enum):
    IN_SOURCE = "inSource"
    EXTERNAL = "external"


class SuppressionStatus(enum.Enum):
    ACCEPTED = "accepted"
    UNDER_REVIEW = "underReview"
    REJECTED = "rejected"


class BaselineState(enum.Enum):
    NEW = "new"
    UNCHANGED = "unchanged"
    UPDATED = "updated"
    ABSENT = "absent"


class ThreadFlowImportance(enum.Enum):
    ESSENTIAL = "essential"
    IMPORTANT = "important"
    UNIMPORTANT = "unimportant"


class ColumnKind(enum.Enum):
    UTF16_CODE_UNITS = "utf16CodeUnits"
    UNICODE_CODE_POINTS = "unicodeCodePoints"


# ---------------------------------------------------------------------------
# 1. SarifPropertyBag
# ---------------------------------------------------------------------------


@dataclass
class SarifPropertyBag:
    """Extensible key-value properties dict wrapper."""

    _properties: Dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self._properties.get(key, default)

    def set(self, key: str, value: Any) -> None:
        self._properties[key] = value

    def has(self, key: str) -> bool:
        return key in self._properties

    def remove(self, key: str) -> bool:
        if key in self._properties:
            del self._properties[key]
            return True
        return False

    def merge(self, other: SarifPropertyBag) -> None:
        self._properties.update(other._properties)

    def keys(self) -> List[str]:
        return list(self._properties.keys())

    def values(self) -> List[Any]:
        return list(self._properties.values())

    def items(self) -> List[Tuple[str, Any]]:
        return list(self._properties.items())

    def __len__(self) -> int:
        return len(self._properties)

    def __bool__(self) -> bool:
        return bool(self._properties)

    def clear(self) -> None:
        self._properties.clear()

    def copy(self) -> SarifPropertyBag:
        return SarifPropertyBag(_properties=copy.deepcopy(self._properties))

    def to_sarif(self) -> Dict[str, Any]:
        return copy.deepcopy(self._properties)

    def to_json(self) -> str:
        return json.dumps(self._properties, indent=2, default=str)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> SarifPropertyBag:
        return cls(_properties=copy.deepcopy(d))


# ---------------------------------------------------------------------------
# 2. SarifMessage
# ---------------------------------------------------------------------------


@dataclass
class SarifMessage:
    """SARIF message with text, markdown, and parameterized arguments."""

    text: str = ""
    markdown: str = ""
    id: Optional[str] = None
    arguments: List[str] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def resolve_arguments(self) -> str:
        result = self.text
        for idx, arg in enumerate(self.arguments):
            placeholder = "{" + str(idx) + "}"
            result = result.replace(placeholder, arg)
        return result

    def resolve_markdown_arguments(self) -> str:
        result = self.markdown
        for idx, arg in enumerate(self.arguments):
            placeholder = "{" + str(idx) + "}"
            result = result.replace(placeholder, arg)
        return result

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.text:
            obj["text"] = self.text
        if self.markdown:
            obj["markdown"] = self.markdown
        if self.id is not None:
            obj["id"] = self.id
        if self.arguments:
            obj["arguments"] = list(self.arguments)
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def from_text(cls, text: str) -> SarifMessage:
        return cls(text=text)

    @classmethod
    def from_markdown(cls, markdown: str) -> SarifMessage:
        plain = re.sub(r"\*\*(.*?)\*\*", r"\1", markdown)
        plain = re.sub(r"\*(.*?)\*", r"\1", plain)
        plain = re.sub(r"`(.*?)`", r"\1", plain)
        plain = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", plain)
        plain = re.sub(r"#{1,6}\s*", "", plain)
        return cls(text=plain.strip(), markdown=markdown)

    @classmethod
    def from_template(cls, template: str, *args: str) -> SarifMessage:
        return cls(text=template, arguments=list(args))

    def __str__(self) -> str:
        return self.resolve_arguments() if self.arguments else self.text


# ---------------------------------------------------------------------------
# 3. SarifLocation (physical + logical)
# ---------------------------------------------------------------------------


@dataclass
class SarifRegion:
    """A region within an artifact (line/column based)."""

    start_line: int = 1
    start_column: int = 1
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    char_offset: Optional[int] = None
    char_length: Optional[int] = None
    byte_offset: Optional[int] = None
    byte_length: Optional[int] = None
    snippet_text: Optional[str] = None
    message: Optional[SarifMessage] = None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"startLine": self.start_line}
        if self.start_column > 1:
            obj["startColumn"] = self.start_column
        if self.end_line is not None:
            obj["endLine"] = self.end_line
        if self.end_column is not None:
            obj["endColumn"] = self.end_column
        if self.char_offset is not None:
            obj["charOffset"] = self.char_offset
        if self.char_length is not None:
            obj["charLength"] = self.char_length
        if self.byte_offset is not None:
            obj["byteOffset"] = self.byte_offset
        if self.byte_length is not None:
            obj["byteLength"] = self.byte_length
        if self.snippet_text is not None:
            obj["snippet"] = {"text": self.snippet_text}
        if self.message is not None:
            obj["message"] = self.message.to_sarif()
        return obj

    def contains(self, line: int, column: int) -> bool:
        if line < self.start_line:
            return False
        end = self.end_line if self.end_line is not None else self.start_line
        if line > end:
            return False
        if line == self.start_line and column < self.start_column:
            return False
        if self.end_column is not None and line == end and column > self.end_column:
            return False
        return True

    def overlaps(self, other: SarifRegion) -> bool:
        self_end_line = self.end_line if self.end_line is not None else self.start_line
        other_end_line = other.end_line if other.end_line is not None else other.start_line
        if self_end_line < other.start_line or other_end_line < self.start_line:
            return False
        if self_end_line == other.start_line:
            self_end_col = self.end_column if self.end_column is not None else self.start_column
            if self_end_col < other.start_column:
                return False
        if other_end_line == self.start_line:
            other_end_col = other.end_column if other.end_column is not None else other.start_column
            if other_end_col < self.start_column:
                return False
        return True

    def line_count(self) -> int:
        end = self.end_line if self.end_line is not None else self.start_line
        return end - self.start_line + 1


@dataclass
class SarifArtifactLocation:
    """Reference to an artifact by URI."""

    uri: str = ""
    uri_base_id: Optional[str] = None
    index: int = -1
    description: Optional[SarifMessage] = None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.uri:
            obj["uri"] = self.uri
        if self.uri_base_id is not None:
            obj["uriBaseId"] = self.uri_base_id
        if self.index >= 0:
            obj["index"] = self.index
        if self.description is not None:
            obj["description"] = self.description.to_sarif()
        return obj

    def get_filename(self) -> str:
        return Path(self.uri).name

    def get_extension(self) -> str:
        return Path(self.uri).suffix

    def resolve_against(self, base_path: str) -> str:
        if os.path.isabs(self.uri):
            return self.uri
        return os.path.normpath(os.path.join(base_path, self.uri))


@dataclass
class SarifPhysicalLocation:
    """Physical location within a file."""

    artifact_location: SarifArtifactLocation = field(default_factory=SarifArtifactLocation)
    region: Optional[SarifRegion] = None
    context_region: Optional[SarifRegion] = None
    address: Optional[Dict[str, Any]] = None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"artifactLocation": self.artifact_location.to_sarif()}
        if self.region is not None:
            obj["region"] = self.region.to_sarif()
        if self.context_region is not None:
            obj["contextRegion"] = self.context_region.to_sarif()
        if self.address is not None:
            obj["address"] = self.address
        return obj


@dataclass
class SarifLogicalLocation:
    """Logical location (function, class, module, etc.)."""

    name: Optional[str] = None
    fully_qualified_name: Optional[str] = None
    decorated_name: Optional[str] = None
    kind: Optional[LogicalLocationKind] = None
    parent_index: int = -1
    index: int = -1
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.name is not None:
            obj["name"] = self.name
        if self.fully_qualified_name is not None:
            obj["fullyQualifiedName"] = self.fully_qualified_name
        if self.decorated_name is not None:
            obj["decoratedName"] = self.decorated_name
        if self.kind is not None:
            obj["kind"] = self.kind.value
        if self.parent_index >= 0:
            obj["parentIndex"] = self.parent_index
        if self.index >= 0:
            obj["index"] = self.index
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


@dataclass
class SarifLocation:
    """A location combining physical and logical."""

    physical_location: Optional[SarifPhysicalLocation] = None
    logical_locations: List[SarifLogicalLocation] = field(default_factory=list)
    id: int = -1
    message: Optional[SarifMessage] = None
    annotations: List[SarifRegion] = field(default_factory=list)
    relationships: List[Dict[str, Any]] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.physical_location is not None:
            obj["physicalLocation"] = self.physical_location.to_sarif()
        if self.logical_locations:
            obj["logicalLocations"] = [ll.to_sarif() for ll in self.logical_locations]
        if self.id >= 0:
            obj["id"] = self.id
        if self.message is not None:
            obj["message"] = self.message.to_sarif()
        if self.annotations:
            obj["annotations"] = [a.to_sarif() for a in self.annotations]
        if self.relationships:
            obj["relationships"] = self.relationships
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def from_file_line(
        cls,
        uri: str,
        start_line: int,
        start_column: int = 1,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        message: Optional[str] = None,
    ) -> SarifLocation:
        region = SarifRegion(
            start_line=start_line,
            start_column=start_column,
            end_line=end_line,
            end_column=end_column,
        )
        phys = SarifPhysicalLocation(
            artifact_location=SarifArtifactLocation(uri=uri),
            region=region,
        )
        msg = SarifMessage.from_text(message) if message else None
        return cls(physical_location=phys, message=msg)

    def get_uri(self) -> str:
        if self.physical_location and self.physical_location.artifact_location:
            return self.physical_location.artifact_location.uri
        return ""

    def get_start_line(self) -> int:
        if self.physical_location and self.physical_location.region:
            return self.physical_location.region.start_line
        return 0


# ---------------------------------------------------------------------------
# 4. SarifArtifact
# ---------------------------------------------------------------------------


@dataclass
class SarifArtifactContent:
    """Content of an artifact (text or binary)."""

    text: Optional[str] = None
    binary: Optional[str] = None  # base64-encoded
    rendered: Optional[Dict[str, Any]] = None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.text is not None:
            obj["text"] = self.text
        if self.binary is not None:
            obj["binary"] = self.binary
        if self.rendered is not None:
            obj["rendered"] = self.rendered
        return obj

    @classmethod
    def from_text(cls, text: str) -> SarifArtifactContent:
        return cls(text=text)

    @classmethod
    def from_binary(cls, data: bytes) -> SarifArtifactContent:
        return cls(binary=base64.b64encode(data).decode("ascii"))


@dataclass
class SarifArtifact:
    """Represents a file artifact in the analysis."""

    location: SarifArtifactLocation = field(default_factory=SarifArtifactLocation)
    description: Optional[SarifMessage] = None
    mime_type: Optional[str] = None
    hashes: Dict[str, str] = field(default_factory=dict)
    length: int = -1
    encoding: Optional[str] = None
    source_language: Optional[str] = None
    roles: List[ArtifactRole] = field(default_factory=list)
    content: Optional[SarifArtifactContent] = None
    last_modified_time_utc: Optional[str] = None
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def compute_hashes_from_content(self, data: bytes) -> None:
        self.hashes["sha-256"] = hashlib.sha256(data).hexdigest()
        self.hashes["md5"] = hashlib.md5(data).hexdigest()

    def compute_hashes_from_file(self, file_path: str) -> bool:
        try:
            with open(file_path, "rb") as f:
                data = f.read()
            self.compute_hashes_from_content(data)
            self.length = len(data)
            return True
        except OSError:
            return False

    def set_content_from_file(self, file_path: str, max_bytes: int = 4096) -> bool:
        try:
            with open(file_path, "rb") as f:
                data = f.read(max_bytes)
            try:
                text = data.decode("utf-8")
                self.content = SarifArtifactContent(text=text)
            except UnicodeDecodeError:
                self.content = SarifArtifactContent.from_binary(data)
            return True
        except OSError:
            return False

    def guess_mime_type(self) -> str:
        ext_map: Dict[str, str] = {
            ".py": "text/x-python",
            ".js": "text/javascript",
            ".ts": "text/typescript",
            ".rb": "text/x-ruby",
            ".java": "text/x-java-source",
            ".c": "text/x-c",
            ".cpp": "text/x-c++src",
            ".h": "text/x-c",
            ".go": "text/x-go",
            ".rs": "text/x-rustsrc",
            ".json": "application/json",
            ".xml": "text/xml",
            ".yaml": "text/yaml",
            ".yml": "text/yaml",
            ".md": "text/markdown",
            ".txt": "text/plain",
            ".html": "text/html",
            ".css": "text/css",
            ".lua": "text/x-lua",
            ".php": "text/x-php",
            ".sh": "text/x-shellscript",
        }
        ext = self.location.get_extension().lower()
        guessed = ext_map.get(ext, "application/octet-stream")
        if self.mime_type is None:
            self.mime_type = guessed
        return guessed

    def guess_source_language(self) -> str:
        ext_map: Dict[str, str] = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".rb": "ruby",
            ".java": "java",
            ".c": "c",
            ".cpp": "cplusplus",
            ".go": "go",
            ".rs": "rust",
            ".lua": "lua",
            ".php": "php",
        }
        ext = self.location.get_extension().lower()
        lang = ext_map.get(ext, "")
        if lang and self.source_language is None:
            self.source_language = lang
        return lang

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"location": self.location.to_sarif()}
        if self.description is not None:
            obj["description"] = self.description.to_sarif()
        if self.mime_type is not None:
            obj["mimeType"] = self.mime_type
        if self.hashes:
            obj["hashes"] = dict(self.hashes)
        if self.length >= 0:
            obj["length"] = self.length
        if self.encoding is not None:
            obj["encoding"] = self.encoding
        if self.source_language is not None:
            obj["sourceLanguage"] = self.source_language
        if self.roles:
            obj["roles"] = [r.value for r in self.roles]
        if self.content is not None:
            obj["contents"] = self.content.to_sarif()
        if self.last_modified_time_utc is not None:
            obj["lastModifiedTimeUtc"] = self.last_modified_time_utc
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def from_file(cls, file_path: str, base_uri: str = "") -> SarifArtifact:
        if base_uri and file_path.startswith(base_uri):
            relative = os.path.relpath(file_path, base_uri)
        else:
            relative = file_path
        uri = relative.replace(os.sep, "/")
        artifact = cls(
            location=SarifArtifactLocation(uri=uri),
            roles=[ArtifactRole.ANALYSIS_TARGET],
        )
        artifact.guess_mime_type()
        artifact.guess_source_language()
        artifact.compute_hashes_from_file(file_path)
        return artifact


# ---------------------------------------------------------------------------
# 5. SarifRule
# ---------------------------------------------------------------------------


@dataclass
class SarifRuleConfiguration:
    """Default configuration for a rule."""

    level: SarifLevel = SarifLevel.WARNING
    rank: float = -1.0
    enabled: bool = True
    parameters: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"level": self.level.value}
        if self.rank >= 0:
            obj["rank"] = self.rank
        if not self.enabled:
            obj["enabled"] = False
        if self.parameters:
            obj["parameters"] = self.parameters.to_sarif()
        return obj


@dataclass
class SarifRuleRelationship:
    """Relationship between a rule and a taxonomy entry."""

    target_id: str = ""
    target_index: int = -1
    target_tool_component_index: int = -1
    kinds: List[str] = field(default_factory=list)
    description: Optional[SarifMessage] = None

    def to_sarif(self) -> Dict[str, Any]:
        target: Dict[str, Any] = {"id": self.target_id}
        if self.target_index >= 0:
            target["index"] = self.target_index
        if self.target_tool_component_index >= 0:
            target["toolComponent"] = {"index": self.target_tool_component_index}
        obj: Dict[str, Any] = {"target": target}
        if self.kinds:
            obj["kinds"] = list(self.kinds)
        if self.description is not None:
            obj["description"] = self.description.to_sarif()
        return obj


@dataclass
class SarifRule:
    """Rule definition for a diagnostic check."""

    id: str = ""
    name: str = ""
    short_description: Optional[SarifMessage] = None
    full_description: Optional[SarifMessage] = None
    help_uri: str = ""
    help: Optional[SarifMessage] = None
    default_configuration: SarifRuleConfiguration = field(
        default_factory=SarifRuleConfiguration
    )
    relationships: List[SarifRuleRelationship] = field(default_factory=list)
    deprecated_ids: List[str] = field(default_factory=list)
    deprecated_names: List[str] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"id": self.id}
        if self.name:
            obj["name"] = self.name
        if self.short_description is not None:
            obj["shortDescription"] = self.short_description.to_sarif()
        if self.full_description is not None:
            obj["fullDescription"] = self.full_description.to_sarif()
        if self.help_uri:
            obj["helpUri"] = self.help_uri
        if self.help is not None:
            obj["help"] = self.help.to_sarif()
        obj["defaultConfiguration"] = self.default_configuration.to_sarif()
        if self.relationships:
            obj["relationships"] = [r.to_sarif() for r in self.relationships]
        if self.deprecated_ids:
            obj["deprecatedIds"] = list(self.deprecated_ids)
        if self.deprecated_names:
            obj["deprecatedNames"] = list(self.deprecated_names)
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


def _create_builtin_rules() -> List[SarifRule]:
    """Create the standard refinement-type rules."""
    rule_defs: List[Tuple[str, str, str, str, SarifLevel, str, float, str]] = [
        (
            "RT001",
            "array-oob",
            "Array index out of bounds",
            "The refinement type analysis determined that an array index may be "
            "outside the valid range of the array. The inferred index type does "
            "not satisfy the length constraint of the array.",
            SarifLevel.ERROR,
            "CWE-125",
            90.0,
            "9.8",
        ),
        (
            "RT002",
            "null-deref",
            "Null pointer dereference",
            "A value with a refinement type that includes null is dereferenced "
            "without a prior null check. The type system cannot prove the value "
            "is non-null at the point of use.",
            SarifLevel.ERROR,
            "CWE-476",
            85.0,
            "7.5",
        ),
        (
            "RT003",
            "div-by-zero",
            "Division by zero",
            "The divisor expression has a refinement type that includes zero. "
            "The type system cannot prove the divisor is non-zero at the "
            "division site.",
            SarifLevel.ERROR,
            "CWE-369",
            80.0,
            "7.5",
        ),
        (
            "RT004",
            "type-confusion",
            "Type confusion",
            "A value is used in a context that requires a different refinement "
            "type. The inferred type is incompatible with the expected type at "
            "the point of use, indicating a potential type confusion bug.",
            SarifLevel.ERROR,
            "CWE-843",
            75.0,
            "8.1",
        ),
        (
            "RT005",
            "unreachable-code",
            "Unreachable code detected",
            "Refinement type analysis determined that a code path is unreachable "
            "because the type constraints at that point are unsatisfiable. This "
            "may indicate dead code or a logic error.",
            SarifLevel.WARNING,
            "",
            60.0,
            "",
        ),
        (
            "RT006",
            "unused-refinement",
            "Unused refinement",
            "A refinement type annotation or constraint is defined but never "
            "used in the analysis. This may indicate stale annotations that "
            "should be removed or updated.",
            SarifLevel.NOTE,
            "",
            30.0,
            "",
        ),
        (
            "RT007",
            "contract-violation",
            "Contract violation",
            "A function precondition or postcondition expressed as a refinement "
            "type is violated. The caller does not satisfy the function's "
            "precondition type, or the function body does not establish its "
            "postcondition type.",
            SarifLevel.ERROR,
            "",
            80.0,
            "",
        ),
        (
            "RT008",
            "unchecked-cast",
            "Unchecked type cast",
            "A type cast is performed without verifying that the runtime value "
            "matches the target refinement type. The type system cannot "
            "statically prove the cast is safe.",
            SarifLevel.WARNING,
            "CWE-843",
            65.0,
            "6.5",
        ),
    ]

    rules: List[SarifRule] = []
    for rule_id, name, short_desc, full_desc, level, cwe, rank, sec_sev in rule_defs:
        props = SarifPropertyBag()
        tags: List[str] = ["refinement-types"]
        if cwe:
            tags.append("security")
            tags.append(f"external/cwe/{cwe.lower()}")
        if sec_sev:
            props.set("security-severity", sec_sev)
        props.set("tags", tags)

        relationships: List[SarifRuleRelationship] = []
        if cwe:
            cwe_num = cwe.split("-")[1]
            relationships.append(
                SarifRuleRelationship(
                    target_id=cwe,
                    kinds=["superset"],
                    description=SarifMessage.from_text(
                        f"Maps to {cwe}: a common weakness enumeration entry."
                    ),
                )
            )

        rule = SarifRule(
            id=rule_id,
            name=name,
            short_description=SarifMessage.from_text(short_desc),
            full_description=SarifMessage.from_text(full_desc),
            help_uri=f"{TOOL_INFO_URI}/rules/{rule_id}",
            help=SarifMessage.from_markdown(
                f"# {short_desc}\n\n{full_desc}\n\n"
                f"## How to fix\n\nAdd appropriate type guards or "
                f"refinement annotations to prove the property at the use site."
            ),
            default_configuration=SarifRuleConfiguration(
                level=level, rank=rank, enabled=True
            ),
            relationships=relationships,
            properties=props,
        )
        rules.append(rule)
    return rules


BUILTIN_RULES: List[SarifRule] = _create_builtin_rules()

RULE_ID_TO_INDEX: Dict[str, int] = {r.id: i for i, r in enumerate(BUILTIN_RULES)}
RULE_NAME_TO_ID: Dict[str, str] = {r.name: r.id for r in BUILTIN_RULES}


def get_rule_by_id(rule_id: str) -> Optional[SarifRule]:
    idx = RULE_ID_TO_INDEX.get(rule_id)
    if idx is not None:
        return BUILTIN_RULES[idx]
    return None


def get_rule_by_name(name: str) -> Optional[SarifRule]:
    rid = RULE_NAME_TO_ID.get(name)
    if rid is not None:
        return get_rule_by_id(rid)
    return None


# ---------------------------------------------------------------------------
# 6. SarifFix
# ---------------------------------------------------------------------------


@dataclass
class SarifReplacement:
    """A single text replacement within an artifact."""

    deleted_region: SarifRegion = field(default_factory=SarifRegion)
    inserted_content: Optional[SarifArtifactContent] = None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"deletedRegion": self.deleted_region.to_sarif()}
        if self.inserted_content is not None:
            obj["insertedContent"] = self.inserted_content.to_sarif()
        return obj

    @classmethod
    def insert_at(cls, line: int, column: int, text: str) -> SarifReplacement:
        return cls(
            deleted_region=SarifRegion(
                start_line=line,
                start_column=column,
                end_line=line,
                end_column=column,
            ),
            inserted_content=SarifArtifactContent(text=text),
        )

    @classmethod
    def replace_range(
        cls,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        new_text: str,
    ) -> SarifReplacement:
        return cls(
            deleted_region=SarifRegion(
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
            ),
            inserted_content=SarifArtifactContent(text=new_text),
        )

    @classmethod
    def delete_range(
        cls, start_line: int, start_column: int, end_line: int, end_column: int
    ) -> SarifReplacement:
        return cls(
            deleted_region=SarifRegion(
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
            )
        )


@dataclass
class SarifArtifactChange:
    """Set of replacements for a single artifact."""

    artifact_location: SarifArtifactLocation = field(
        default_factory=SarifArtifactLocation
    )
    replacements: List[SarifReplacement] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def add_replacement(self, replacement: SarifReplacement) -> None:
        self.replacements.append(replacement)

    def sort_replacements(self) -> None:
        self.replacements.sort(
            key=lambda r: (r.deleted_region.start_line, r.deleted_region.start_column),
            reverse=True,
        )

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "artifactLocation": self.artifact_location.to_sarif(),
            "replacements": [r.to_sarif() for r in self.replacements],
        }
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


@dataclass
class SarifFix:
    """A proposed fix for a diagnostic result."""

    description: SarifMessage = field(default_factory=lambda: SarifMessage.from_text(""))
    artifact_changes: List[SarifArtifactChange] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def add_change(self, change: SarifArtifactChange) -> None:
        self.artifact_changes.append(change)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"description": self.description.to_sarif()}
        if self.artifact_changes:
            obj["artifactChanges"] = [c.to_sarif() for c in self.artifact_changes]
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def simple_replacement(
        cls,
        uri: str,
        start_line: int,
        start_column: int,
        end_line: int,
        end_column: int,
        new_text: str,
        description: str = "Apply suggested fix",
    ) -> SarifFix:
        replacement = SarifReplacement.replace_range(
            start_line, start_column, end_line, end_column, new_text
        )
        change = SarifArtifactChange(
            artifact_location=SarifArtifactLocation(uri=uri),
            replacements=[replacement],
        )
        return cls(
            description=SarifMessage.from_text(description),
            artifact_changes=[change],
        )

    @classmethod
    def insert_guard(
        cls,
        uri: str,
        line: int,
        column: int,
        guard_text: str,
        description: str = "Insert type guard",
    ) -> SarifFix:
        replacement = SarifReplacement.insert_at(line, column, guard_text)
        change = SarifArtifactChange(
            artifact_location=SarifArtifactLocation(uri=uri),
            replacements=[replacement],
        )
        return cls(
            description=SarifMessage.from_text(description),
            artifact_changes=[change],
        )


# ---------------------------------------------------------------------------
# 7. SarifThreadFlow
# ---------------------------------------------------------------------------


@dataclass
class SarifThreadFlowLocation:
    """A single location within a thread flow (bug path step)."""

    location: Optional[SarifLocation] = None
    state: Dict[str, str] = field(default_factory=dict)
    nesting_level: int = 0
    execution_order: int = -1
    importance: ThreadFlowImportance = ThreadFlowImportance.IMPORTANT
    kinds: List[str] = field(default_factory=list)
    taxa: List[Dict[str, Any]] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.location is not None:
            obj["location"] = self.location.to_sarif()
        if self.state:
            obj["state"] = {
                k: {"text": v} for k, v in self.state.items()
            }
        if self.nesting_level > 0:
            obj["nestingLevel"] = self.nesting_level
        if self.execution_order >= 0:
            obj["executionOrder"] = self.execution_order
        obj["importance"] = self.importance.value
        if self.kinds:
            obj["kinds"] = list(self.kinds)
        if self.taxa:
            obj["taxa"] = self.taxa
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


@dataclass
class SarifThreadFlow:
    """A sequence of code locations forming a thread of execution."""

    id: Optional[str] = None
    message: Optional[SarifMessage] = None
    locations: List[SarifThreadFlowLocation] = field(default_factory=list)
    initial_state: Dict[str, str] = field(default_factory=dict)
    immutable_state: Dict[str, str] = field(default_factory=dict)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def add_location(self, tfl: SarifThreadFlowLocation) -> None:
        if tfl.execution_order < 0:
            tfl.execution_order = len(self.locations) + 1
        self.locations.append(tfl)

    def add_step(
        self,
        uri: str,
        line: int,
        column: int = 1,
        message: str = "",
        importance: ThreadFlowImportance = ThreadFlowImportance.IMPORTANT,
        state: Optional[Dict[str, str]] = None,
        nesting_level: int = 0,
    ) -> SarifThreadFlowLocation:
        loc = SarifLocation.from_file_line(uri, line, column, message=message)
        tfl = SarifThreadFlowLocation(
            location=loc,
            state=state or {},
            nesting_level=nesting_level,
            importance=importance,
        )
        self.add_location(tfl)
        return tfl

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "locations": [loc.to_sarif() for loc in self.locations]
        }
        if self.id is not None:
            obj["id"] = self.id
        if self.message is not None:
            obj["message"] = self.message.to_sarif()
        if self.initial_state:
            obj["initialState"] = {
                k: {"text": v} for k, v in self.initial_state.items()
            }
        if self.immutable_state:
            obj["immutableState"] = {
                k: {"text": v} for k, v in self.immutable_state.items()
            }
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 8. SarifCodeFlow
# ---------------------------------------------------------------------------


@dataclass
class SarifCodeFlow:
    """A code flow representing a path through the program."""

    message: Optional[SarifMessage] = None
    thread_flows: List[SarifThreadFlow] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def add_thread_flow(self, tf: SarifThreadFlow) -> None:
        self.thread_flows.append(tf)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "threadFlows": [tf.to_sarif() for tf in self.thread_flows]
        }
        if self.message is not None:
            obj["message"] = self.message.to_sarif()
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def from_path(
        cls, message: str, steps: List[Tuple[str, int, int, str]]
    ) -> SarifCodeFlow:
        tf = SarifThreadFlow(message=SarifMessage.from_text(message))
        for uri, line, col, step_msg in steps:
            tf.add_step(uri, line, col, message=step_msg)
        return cls(
            message=SarifMessage.from_text(message),
            thread_flows=[tf],
        )

    @classmethod
    def null_deref_path(
        cls,
        assign_uri: str,
        assign_line: int,
        deref_uri: str,
        deref_line: int,
        var_name: str = "x",
    ) -> SarifCodeFlow:
        return cls.from_path(
            f"Variable '{var_name}' may be null when dereferenced",
            [
                (assign_uri, assign_line, 1, f"'{var_name}' assigned null here"),
                (deref_uri, deref_line, 1, f"'{var_name}' dereferenced here"),
            ],
        )


# ---------------------------------------------------------------------------
# 9. SarifFingerprint
# ---------------------------------------------------------------------------


@dataclass
class SarifFingerprint:
    """Generates stable fingerprints for results."""

    fingerprints: Dict[str, str] = field(default_factory=dict)
    partial_fingerprints: Dict[str, str] = field(default_factory=dict)
    correlation_guid: str = ""

    def compute_fingerprint_v1(
        self,
        rule_id: str,
        file_uri: str,
        function_name: str = "",
    ) -> str:
        data = f"{rule_id}|{file_uri}|{function_name}"
        fp = hashlib.sha256(data.encode("utf-8")).hexdigest()
        self.fingerprints["v1"] = fp
        return fp

    def compute_fingerprint_v2(
        self,
        rule_id: str,
        file_uri: str,
        code_context: str = "",
        line_offset_in_function: int = 0,
    ) -> str:
        normalized = re.sub(r"\s+", " ", code_context.strip())
        data = f"{rule_id}|{file_uri}|{normalized}|{line_offset_in_function}"
        fp = hashlib.sha256(data.encode("utf-8")).hexdigest()
        self.fingerprints["v2"] = fp
        return fp

    def compute_partial_fingerprint(
        self,
        key: str,
        *components: str,
    ) -> str:
        data = "|".join(components)
        fp = hashlib.sha256(data.encode("utf-8")).hexdigest()[:32]
        self.partial_fingerprints[key] = fp
        return fp

    def generate_correlation_guid(self) -> str:
        if self.fingerprints:
            seed = next(iter(self.fingerprints.values()))
            self.correlation_guid = str(uuid.uuid5(uuid.NAMESPACE_URL, seed))
        else:
            self.correlation_guid = str(uuid.uuid4())
        return self.correlation_guid

    def to_fingerprints_dict(self) -> Dict[str, str]:
        return dict(self.fingerprints)

    def to_partial_fingerprints_dict(self) -> Dict[str, str]:
        return dict(self.partial_fingerprints)


# ---------------------------------------------------------------------------
# 10. SarifResult
# ---------------------------------------------------------------------------


@dataclass
class SarifResult:
    """An individual diagnostic result."""

    rule_id: str = ""
    rule_index: int = -1
    level: SarifLevel = SarifLevel.WARNING
    message: SarifMessage = field(default_factory=lambda: SarifMessage.from_text(""))
    locations: List[SarifLocation] = field(default_factory=list)
    related_locations: List[SarifLocation] = field(default_factory=list)
    code_flows: List[SarifCodeFlow] = field(default_factory=list)
    fixes: List[SarifFix] = field(default_factory=list)
    fingerprints: Dict[str, str] = field(default_factory=dict)
    partial_fingerprints: Dict[str, str] = field(default_factory=dict)
    correlation_guid: str = ""
    guid: str = ""
    occurrence_count: int = 1
    rank: float = -1.0
    baseline_state: Optional[BaselineState] = None
    suppressions: List[SarifSuppression] = field(default_factory=list)
    stacks: List[Dict[str, Any]] = field(default_factory=list)
    graphs: List[SarifGraph] = field(default_factory=list)
    graph_traversals: List[Dict[str, Any]] = field(default_factory=list)
    work_item_uris: List[str] = field(default_factory=list)
    hosted_viewer_uri: str = ""
    provenance: Dict[str, Any] = field(default_factory=dict)
    taxa: List[Dict[str, Any]] = field(default_factory=list)
    web_request: Optional[Dict[str, Any]] = None
    web_response: Optional[Dict[str, Any]] = None
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def __post_init__(self) -> None:
        if not self.guid:
            self.guid = str(uuid.uuid4())

    def add_location(self, loc: SarifLocation) -> None:
        self.locations.append(loc)

    def add_related_location(self, loc: SarifLocation) -> None:
        loc.id = len(self.related_locations)
        self.related_locations.append(loc)

    def add_code_flow(self, cf: SarifCodeFlow) -> None:
        self.code_flows.append(cf)

    def add_fix(self, fix: SarifFix) -> None:
        self.fixes.append(fix)

    def compute_fingerprints(
        self,
        function_name: str = "",
        code_context: str = "",
        line_offset: int = 0,
    ) -> None:
        fp = SarifFingerprint()
        uri = ""
        if self.locations:
            uri = self.locations[0].get_uri()
        fp.compute_fingerprint_v1(self.rule_id, uri, function_name)
        if code_context:
            fp.compute_fingerprint_v2(self.rule_id, uri, code_context, line_offset)
        fp.compute_partial_fingerprint(
            "primaryLocationLineHash", self.rule_id, uri, str(self.get_primary_line())
        )
        fp.generate_correlation_guid()
        self.fingerprints = fp.to_fingerprints_dict()
        self.partial_fingerprints = fp.to_partial_fingerprints_dict()
        self.correlation_guid = fp.correlation_guid

    def get_primary_line(self) -> int:
        if self.locations:
            return self.locations[0].get_start_line()
        return 0

    def get_primary_uri(self) -> str:
        if self.locations:
            return self.locations[0].get_uri()
        return ""

    def is_suppressed(self) -> bool:
        return any(
            s.status == SuppressionStatus.ACCEPTED for s in self.suppressions
        )

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "ruleId": self.rule_id,
            "level": self.level.value,
            "message": self.message.to_sarif(),
        }
        if self.rule_index >= 0:
            obj["ruleIndex"] = self.rule_index
        if self.locations:
            obj["locations"] = [loc.to_sarif() for loc in self.locations]
        if self.related_locations:
            obj["relatedLocations"] = [
                loc.to_sarif() for loc in self.related_locations
            ]
        if self.code_flows:
            obj["codeFlows"] = [cf.to_sarif() for cf in self.code_flows]
        if self.fixes:
            obj["fixes"] = [f.to_sarif() for f in self.fixes]
        if self.fingerprints:
            obj["fingerprints"] = dict(self.fingerprints)
        if self.partial_fingerprints:
            obj["partialFingerprints"] = dict(self.partial_fingerprints)
        if self.correlation_guid:
            obj["correlationGuid"] = self.correlation_guid
        if self.guid:
            obj["guid"] = self.guid
        if self.occurrence_count > 1:
            obj["occurrenceCount"] = self.occurrence_count
        if self.rank >= 0:
            obj["rank"] = self.rank
        if self.baseline_state is not None:
            obj["baselineState"] = self.baseline_state.value
        if self.suppressions:
            obj["suppressions"] = [s.to_sarif() for s in self.suppressions]
        if self.stacks:
            obj["stacks"] = self.stacks
        if self.graphs:
            obj["graphs"] = [g.to_sarif() for g in self.graphs]
        if self.graph_traversals:
            obj["graphTraversals"] = self.graph_traversals
        if self.work_item_uris:
            obj["workItemUris"] = list(self.work_item_uris)
        if self.hosted_viewer_uri:
            obj["hostedViewerUri"] = self.hosted_viewer_uri
        if self.provenance:
            obj["provenance"] = self.provenance
        if self.taxa:
            obj["taxa"] = self.taxa
        if self.web_request is not None:
            obj["webRequest"] = self.web_request
        if self.web_response is not None:
            obj["webResponse"] = self.web_response
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 11. SarifToolComponent
# ---------------------------------------------------------------------------


@dataclass
class SarifToolComponent:
    """Tool component metadata (driver or extension)."""

    name: str = TOOL_NAME
    version: str = TOOL_VERSION
    semantic_version: str = TOOL_SEMANTIC_VERSION
    information_uri: str = TOOL_INFO_URI
    download_uri: str = ""
    organization: str = TOOL_ORG
    product: str = ""
    product_suite: str = ""
    full_name: str = ""
    short_description: Optional[SarifMessage] = None
    full_description: Optional[SarifMessage] = None
    language: str = "en-US"
    rules: List[SarifRule] = field(default_factory=list)
    notifications: List[Dict[str, Any]] = field(default_factory=list)
    taxa: List[Dict[str, Any]] = field(default_factory=list)
    guid: str = ""
    contents: List[str] = field(default_factory=lambda: ["localizedData", "nonLocalizedData"])
    is_comprehensive: bool = False
    locations: List[SarifArtifactLocation] = field(default_factory=list)
    associated_component: Optional[Dict[str, Any]] = None
    translation_metadata: Optional[Dict[str, Any]] = None
    supported_taxonomies: List[Dict[str, Any]] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def __post_init__(self) -> None:
        if not self.guid:
            self.guid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{self.name}/{self.version}"))

    def add_rule(self, rule: SarifRule) -> int:
        index = len(self.rules)
        self.rules.append(rule)
        return index

    def get_rule_index(self, rule_id: str) -> int:
        for i, rule in enumerate(self.rules):
            if rule.id == rule_id:
                return i
        return -1

    def find_rule(self, rule_id: str) -> Optional[SarifRule]:
        for rule in self.rules:
            if rule.id == rule_id:
                return rule
        return None

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"name": self.name}
        if self.version:
            obj["version"] = self.version
        if self.semantic_version:
            obj["semanticVersion"] = self.semantic_version
        if self.information_uri:
            obj["informationUri"] = self.information_uri
        if self.download_uri:
            obj["downloadUri"] = self.download_uri
        if self.organization:
            obj["organization"] = self.organization
        if self.product:
            obj["product"] = self.product
        if self.product_suite:
            obj["productSuite"] = self.product_suite
        if self.full_name:
            obj["fullName"] = self.full_name
        if self.short_description is not None:
            obj["shortDescription"] = self.short_description.to_sarif()
        if self.full_description is not None:
            obj["fullDescription"] = self.full_description.to_sarif()
        if self.language:
            obj["language"] = self.language
        if self.rules:
            obj["rules"] = [r.to_sarif() for r in self.rules]
        if self.notifications:
            obj["notifications"] = self.notifications
        if self.taxa:
            obj["taxa"] = self.taxa
        if self.guid:
            obj["guid"] = self.guid
        if self.supported_taxonomies:
            obj["supportedTaxonomies"] = self.supported_taxonomies
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 12. SarifInvocation
# ---------------------------------------------------------------------------


_SENSITIVE_ENV_PATTERNS: List[str] = [
    "password", "secret", "token", "key", "credential", "auth",
    "private", "api_key", "apikey", "access_key",
]


def _filter_env_vars(env: Dict[str, str]) -> Dict[str, str]:
    filtered: Dict[str, str] = {}
    for k, v in env.items():
        lower_k = k.lower()
        is_sensitive = any(pat in lower_k for pat in _SENSITIVE_ENV_PATTERNS)
        if is_sensitive:
            filtered[k] = "**REDACTED**"
        else:
            filtered[k] = v
    return filtered


@dataclass
class SarifInvocation:
    """Invocation details for the analysis tool."""

    command_line: str = ""
    arguments: List[str] = field(default_factory=list)
    working_directory: Optional[SarifArtifactLocation] = None
    environment_variables: Dict[str, str] = field(default_factory=dict)
    start_time_utc: str = ""
    end_time_utc: str = ""
    execution_successful: bool = True
    exit_code: int = 0
    exit_code_description: str = ""
    machine: str = ""
    account: str = ""
    process_id: int = -1
    executable_location: Optional[SarifArtifactLocation] = None
    response_files: List[SarifArtifactLocation] = field(default_factory=list)
    stdin: Optional[SarifArtifactLocation] = None
    stdout: Optional[SarifArtifactLocation] = None
    stderr: Optional[SarifArtifactLocation] = None
    stdout_stderr: Optional[SarifArtifactLocation] = None
    tool_execution_notifications: List[SarifNotification] = field(
        default_factory=list
    )
    tool_configuration_notifications: List[SarifNotification] = field(
        default_factory=list
    )
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def start(self) -> None:
        self.start_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        self.machine = os.uname().nodename if hasattr(os, "uname") else ""
        self.account = os.environ.get("USER", os.environ.get("USERNAME", ""))
        self.process_id = os.getpid()

    def finish(self, success: bool = True, exit_code: int = 0) -> None:
        self.end_time_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
        self.execution_successful = success
        self.exit_code = exit_code

    def add_notification(self, notification: SarifNotification) -> None:
        self.tool_execution_notifications.append(notification)

    def add_config_notification(self, notification: SarifNotification) -> None:
        self.tool_configuration_notifications.append(notification)

    def set_working_directory(self, path: str) -> None:
        uri = path.replace(os.sep, "/")
        if not uri.endswith("/"):
            uri += "/"
        self.working_directory = SarifArtifactLocation(uri=uri)

    def capture_environment(self, env: Optional[Dict[str, str]] = None) -> None:
        raw = env if env is not None else dict(os.environ)
        self.environment_variables = _filter_env_vars(raw)

    def duration_seconds(self) -> float:
        if not self.start_time_utc or not self.end_time_utc:
            return 0.0
        fmt = "%Y-%m-%dT%H:%M:%S.%fZ"
        try:
            start = datetime.datetime.strptime(self.start_time_utc, fmt)
            end = datetime.datetime.strptime(self.end_time_utc, fmt)
            return (end - start).total_seconds()
        except ValueError:
            return 0.0

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"executionSuccessful": self.execution_successful}
        if self.command_line:
            obj["commandLine"] = self.command_line
        if self.arguments:
            obj["arguments"] = list(self.arguments)
        if self.working_directory is not None:
            obj["workingDirectory"] = self.working_directory.to_sarif()
        if self.environment_variables:
            obj["environmentVariables"] = dict(self.environment_variables)
        if self.start_time_utc:
            obj["startTimeUtc"] = self.start_time_utc
        if self.end_time_utc:
            obj["endTimeUtc"] = self.end_time_utc
        obj["exitCode"] = self.exit_code
        if self.exit_code_description:
            obj["exitCodeDescription"] = self.exit_code_description
        if self.machine:
            obj["machine"] = self.machine
        if self.account:
            obj["account"] = self.account
        if self.process_id >= 0:
            obj["processId"] = self.process_id
        if self.executable_location is not None:
            obj["executableLocation"] = self.executable_location.to_sarif()
        if self.tool_execution_notifications:
            obj["toolExecutionNotifications"] = [
                n.to_sarif() for n in self.tool_execution_notifications
            ]
        if self.tool_configuration_notifications:
            obj["toolConfigurationNotifications"] = [
                n.to_sarif() for n in self.tool_configuration_notifications
            ]
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 13. SarifNotification
# ---------------------------------------------------------------------------


@dataclass
class SarifExceptionData:
    """Exception details for notifications."""

    kind: str = ""
    message: str = ""
    stack_trace: str = ""
    inner_exceptions: List[SarifExceptionData] = field(default_factory=list)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.kind:
            obj["kind"] = self.kind
        if self.message:
            obj["message"] = self.message
        if self.stack_trace:
            obj["stack"] = {
                "message": {"text": self.stack_trace},
                "frames": self._parse_stack_frames(),
            }
        if self.inner_exceptions:
            obj["innerExceptions"] = [ie.to_sarif() for ie in self.inner_exceptions]
        return obj

    def _parse_stack_frames(self) -> List[Dict[str, Any]]:
        frames: List[Dict[str, Any]] = []
        pattern = re.compile(
            r'File "([^"]+)", line (\d+), in (\S+)'
        )
        for match in pattern.finditer(self.stack_trace):
            filepath, line_str, func = match.groups()
            frame: Dict[str, Any] = {
                "location": {
                    "physicalLocation": {
                        "artifactLocation": {"uri": filepath.replace(os.sep, "/")},
                        "region": {"startLine": int(line_str)},
                    },
                    "logicalLocations": [
                        {"name": func, "kind": "function"}
                    ],
                }
            }
            frames.append(frame)
        return frames


@dataclass
class SarifNotification:
    """A notification from the analysis tool."""

    descriptor_id: str = ""
    descriptor_index: int = -1
    message: SarifMessage = field(default_factory=lambda: SarifMessage.from_text(""))
    level: SarifLevel = SarifLevel.NOTE
    associated_rule_id: str = ""
    associated_rule_index: int = -1
    exception_data: Optional[SarifExceptionData] = None
    time_utc: str = ""
    thread_id: int = -1
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def __post_init__(self) -> None:
        if not self.time_utc:
            self.time_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S.%fZ"
            )

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "message": self.message.to_sarif(),
            "level": self.level.value,
        }
        if self.descriptor_id:
            descriptor: Dict[str, Any] = {"id": self.descriptor_id}
            if self.descriptor_index >= 0:
                descriptor["index"] = self.descriptor_index
            obj["descriptor"] = descriptor
        if self.associated_rule_id:
            assoc: Dict[str, Any] = {"id": self.associated_rule_id}
            if self.associated_rule_index >= 0:
                assoc["index"] = self.associated_rule_index
            obj["associatedRule"] = assoc
        if self.exception_data is not None:
            obj["exception"] = self.exception_data.to_sarif()
        if self.time_utc:
            obj["timeUtc"] = self.time_utc
        if self.thread_id >= 0:
            obj["threadId"] = self.thread_id
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def info(cls, descriptor_id: str, message: str) -> SarifNotification:
        return cls(
            descriptor_id=descriptor_id,
            message=SarifMessage.from_text(message),
            level=SarifLevel.NOTE,
        )

    @classmethod
    def warning(cls, descriptor_id: str, message: str) -> SarifNotification:
        return cls(
            descriptor_id=descriptor_id,
            message=SarifMessage.from_text(message),
            level=SarifLevel.WARNING,
        )

    @classmethod
    def error_with_exception(
        cls,
        descriptor_id: str,
        message: str,
        exception_kind: str = "",
        exception_message: str = "",
        stack_trace: str = "",
    ) -> SarifNotification:
        exc = SarifExceptionData(
            kind=exception_kind,
            message=exception_message,
            stack_trace=stack_trace,
        )
        return cls(
            descriptor_id=descriptor_id,
            message=SarifMessage.from_text(message),
            level=SarifLevel.ERROR,
            exception_data=exc,
        )


# ---------------------------------------------------------------------------
# 14. SarifGraph
# ---------------------------------------------------------------------------


@dataclass
class SarifGraphNode:
    """A node in a SARIF graph."""

    id: str = ""
    label: Optional[SarifMessage] = None
    location: Optional[SarifLocation] = None
    children: List[SarifGraphNode] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"id": self.id}
        if self.label is not None:
            obj["label"] = self.label.to_sarif()
        if self.location is not None:
            obj["location"] = self.location.to_sarif()
        if self.children:
            obj["children"] = [c.to_sarif() for c in self.children]
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


@dataclass
class SarifGraphEdge:
    """An edge in a SARIF graph."""

    id: str = ""
    source_node_id: str = ""
    target_node_id: str = ""
    label: Optional[SarifMessage] = None
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "id": self.id,
            "sourceNodeId": self.source_node_id,
            "targetNodeId": self.target_node_id,
        }
        if self.label is not None:
            obj["label"] = self.label.to_sarif()
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


@dataclass
class SarifGraph:
    """A graph representation (e.g., call graph)."""

    description: Optional[SarifMessage] = None
    nodes: List[SarifGraphNode] = field(default_factory=list)
    edges: List[SarifGraphEdge] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def add_node(
        self,
        node_id: str,
        label: str = "",
        location: Optional[SarifLocation] = None,
    ) -> SarifGraphNode:
        node = SarifGraphNode(
            id=node_id,
            label=SarifMessage.from_text(label) if label else None,
            location=location,
        )
        self.nodes.append(node)
        return node

    def add_edge(
        self,
        edge_id: str,
        source_id: str,
        target_id: str,
        label: str = "",
    ) -> SarifGraphEdge:
        edge = SarifGraphEdge(
            id=edge_id,
            source_node_id=source_id,
            target_node_id=target_id,
            label=SarifMessage.from_text(label) if label else None,
        )
        self.edges.append(edge)
        return edge

    def find_node(self, node_id: str) -> Optional[SarifGraphNode]:
        for node in self.nodes:
            if node.id == node_id:
                return node
        return None

    def get_neighbors(self, node_id: str) -> List[str]:
        neighbors: List[str] = []
        for edge in self.edges:
            if edge.source_node_id == node_id:
                neighbors.append(edge.target_node_id)
            elif edge.target_node_id == node_id:
                neighbors.append(edge.source_node_id)
        return neighbors

    def get_successors(self, node_id: str) -> List[str]:
        return [e.target_node_id for e in self.edges if e.source_node_id == node_id]

    def get_predecessors(self, node_id: str) -> List[str]:
        return [e.source_node_id for e in self.edges if e.target_node_id == node_id]

    def topological_sort(self) -> List[str]:
        in_degree: Dict[str, int] = {n.id: 0 for n in self.nodes}
        adj: Dict[str, List[str]] = {n.id: [] for n in self.nodes}
        for edge in self.edges:
            if edge.source_node_id in adj:
                adj[edge.source_node_id].append(edge.target_node_id)
            if edge.target_node_id in in_degree:
                in_degree[edge.target_node_id] += 1
        queue = collections.deque(
            nid for nid, deg in in_degree.items() if deg == 0
        )
        order: List[str] = []
        while queue:
            nid = queue.popleft()
            order.append(nid)
            for succ in adj.get(nid, []):
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    queue.append(succ)
        return order

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {}
        if self.description is not None:
            obj["description"] = self.description.to_sarif()
        if self.nodes:
            obj["nodes"] = [n.to_sarif() for n in self.nodes]
        if self.edges:
            obj["edges"] = [e.to_sarif() for e in self.edges]
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 15. SarifSuppression
# ---------------------------------------------------------------------------


@dataclass
class SarifSuppression:
    """Suppression tracking for results."""

    kind: SuppressionKind = SuppressionKind.IN_SOURCE
    status: SuppressionStatus = SuppressionStatus.ACCEPTED
    location: Optional[SarifLocation] = None
    justification: str = ""
    guid: str = ""
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def __post_init__(self) -> None:
        if not self.guid:
            self.guid = str(uuid.uuid4())

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "kind": self.kind.value,
            "status": self.status.value,
        }
        if self.location is not None:
            obj["location"] = self.location.to_sarif()
        if self.justification:
            obj["justification"] = self.justification
        if self.guid:
            obj["guid"] = self.guid
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj

    @classmethod
    def in_source(cls, justification: str = "") -> SarifSuppression:
        return cls(
            kind=SuppressionKind.IN_SOURCE,
            status=SuppressionStatus.ACCEPTED,
            justification=justification,
        )

    @classmethod
    def external(
        cls,
        justification: str = "",
        status: SuppressionStatus = SuppressionStatus.ACCEPTED,
    ) -> SarifSuppression:
        return cls(
            kind=SuppressionKind.EXTERNAL,
            status=status,
            justification=justification,
        )


# ---------------------------------------------------------------------------
# 16. SarifTaxonomy
# ---------------------------------------------------------------------------


@dataclass
class SarifTaxon:
    """A single taxon (e.g., CWE entry)."""

    id: str = ""
    name: str = ""
    short_description: Optional[SarifMessage] = None
    full_description: Optional[SarifMessage] = None
    help_uri: str = ""
    default_configuration: Optional[SarifRuleConfiguration] = None
    relationships: List[SarifRuleRelationship] = field(default_factory=list)
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"id": self.id}
        if self.name:
            obj["name"] = self.name
        if self.short_description is not None:
            obj["shortDescription"] = self.short_description.to_sarif()
        if self.full_description is not None:
            obj["fullDescription"] = self.full_description.to_sarif()
        if self.help_uri:
            obj["helpUri"] = self.help_uri
        if self.default_configuration is not None:
            obj["defaultConfiguration"] = self.default_configuration.to_sarif()
        if self.relationships:
            obj["relationships"] = [r.to_sarif() for r in self.relationships]
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


_CWE_ENTRIES: Dict[str, Tuple[str, str, str]] = {
    "CWE-125": (
        "Out-of-bounds Read",
        "The product reads data past the end, or before the beginning, of the intended buffer.",
        "https://cwe.mitre.org/data/definitions/125.html",
    ),
    "CWE-476": (
        "NULL Pointer Dereference",
        "A NULL pointer dereference occurs when the application dereferences a pointer that it expects to be valid, but is NULL.",
        "https://cwe.mitre.org/data/definitions/476.html",
    ),
    "CWE-369": (
        "Divide By Zero",
        "The product divides a value by zero.",
        "https://cwe.mitre.org/data/definitions/369.html",
    ),
    "CWE-843": (
        "Access of Resource Using Incompatible Type (Type Confusion)",
        "The product allocates or initializes a resource such as a pointer, object, or variable using one type, but later accesses that resource using a type that is incompatible with the original type.",
        "https://cwe.mitre.org/data/definitions/843.html",
    ),
}

_RULE_TO_CWE: Dict[str, str] = {
    "RT001": "CWE-125",
    "RT002": "CWE-476",
    "RT003": "CWE-369",
    "RT004": "CWE-843",
    "RT008": "CWE-843",
}


@dataclass
class SarifTaxonomy:
    """CWE taxonomy mapping."""

    name: str = "CWE"
    version: str = "4.13"
    information_uri: str = "https://cwe.mitre.org/data/published/cwe_v4.13.pdf"
    download_uri: str = "https://cwe.mitre.org/data/xml/cwec_v4.13.xml.zip"
    organization: str = "MITRE"
    short_description: Optional[SarifMessage] = None
    is_comprehensive: bool = False
    minimum_required_localized_data_semantic_version: str = ""
    guid: str = ""
    release_date_utc: str = ""
    taxa: List[SarifTaxon] = field(default_factory=list)
    supported_taxonomies: List[Dict[str, Any]] = field(default_factory=list)
    contents: List[str] = field(default_factory=lambda: ["localizedData", "nonLocalizedData"])
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def __post_init__(self) -> None:
        if not self.guid:
            self.guid = str(uuid.uuid5(uuid.NAMESPACE_URL, f"CWE/{self.version}"))
        if not self.short_description:
            self.short_description = SarifMessage.from_text(
                "Common Weakness Enumeration"
            )
        if not self.taxa:
            self._populate_default_taxa()

    def _populate_default_taxa(self) -> None:
        for cwe_id, (name, desc, uri) in _CWE_ENTRIES.items():
            taxon = SarifTaxon(
                id=cwe_id,
                name=name,
                short_description=SarifMessage.from_text(name),
                full_description=SarifMessage.from_text(desc),
                help_uri=uri,
            )
            self.taxa.append(taxon)

    def get_cwe_for_rule(self, rule_id: str) -> Optional[str]:
        return _RULE_TO_CWE.get(rule_id)

    def get_taxon_by_id(self, cwe_id: str) -> Optional[SarifTaxon]:
        for taxon in self.taxa:
            if taxon.id == cwe_id:
                return taxon
        return None

    def get_taxon_index(self, cwe_id: str) -> int:
        for i, taxon in enumerate(self.taxa):
            if taxon.id == cwe_id:
                return i
        return -1

    def add_taxon(self, taxon: SarifTaxon) -> int:
        idx = len(self.taxa)
        self.taxa.append(taxon)
        return idx

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "name": self.name,
            "version": self.version,
            "informationUri": self.information_uri,
            "organization": self.organization,
        }
        if self.download_uri:
            obj["downloadUri"] = self.download_uri
        if self.short_description is not None:
            obj["shortDescription"] = self.short_description.to_sarif()
        if self.guid:
            obj["guid"] = self.guid
        if self.release_date_utc:
            obj["releaseDateUtc"] = self.release_date_utc
        obj["isComprehensive"] = self.is_comprehensive
        if self.taxa:
            obj["taxa"] = [t.to_sarif() for t in self.taxa]
        if self.contents:
            obj["contents"] = self.contents
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 17. SarifBaseline
# ---------------------------------------------------------------------------


@dataclass
class SarifBaseline:
    """Baseline comparison for SARIF results."""

    baseline_results: List[Dict[str, Any]] = field(default_factory=list)
    baseline_fingerprints: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    source_path: str = ""

    def load_baseline(self, file_path: str) -> bool:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.source_path = file_path
            runs = data.get("runs", [])
            if runs:
                self.baseline_results = runs[0].get("results", [])
                for result in self.baseline_results:
                    fps = result.get("fingerprints", {})
                    for fp_key, fp_val in fps.items():
                        self.baseline_fingerprints[fp_val] = result
            return True
        except (OSError, json.JSONDecodeError, KeyError):
            return False

    def save_baseline(self, sarif_data: Dict[str, Any], file_path: str) -> bool:
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sarif_data, f, indent=2, default=str)
            return True
        except OSError:
            return False

    def classify_result(self, result: SarifResult) -> BaselineState:
        if not self.baseline_fingerprints:
            return BaselineState.NEW

        for fp_val in result.fingerprints.values():
            if fp_val in self.baseline_fingerprints:
                baseline_entry = self.baseline_fingerprints[fp_val]
                bl_level = baseline_entry.get("level", "")
                bl_msg = baseline_entry.get("message", {}).get("text", "")
                if bl_level == result.level.value and bl_msg == str(result.message):
                    return BaselineState.UNCHANGED
                return BaselineState.UPDATED
        return BaselineState.NEW

    def compute_absent(
        self, current_results: List[SarifResult]
    ) -> List[Dict[str, Any]]:
        current_fps: Set[str] = set()
        for result in current_results:
            current_fps.update(result.fingerprints.values())

        absent: List[Dict[str, Any]] = []
        for fp_val, baseline_entry in self.baseline_fingerprints.items():
            if fp_val not in current_fps:
                entry = copy.deepcopy(baseline_entry)
                entry["baselineState"] = BaselineState.ABSENT.value
                absent.append(entry)
        return absent

    def apply_baseline(self, results: List[SarifResult]) -> None:
        for result in results:
            result.baseline_state = self.classify_result(result)

    def get_stats(
        self, results: List[SarifResult]
    ) -> Dict[str, int]:
        stats: Dict[str, int] = {
            "new": 0,
            "unchanged": 0,
            "updated": 0,
            "absent": 0,
        }
        for result in results:
            if result.baseline_state is not None:
                stats[result.baseline_state.value] += 1
        absent_list = self.compute_absent(results)
        stats["absent"] = len(absent_list)
        return stats


# ---------------------------------------------------------------------------
# 18. SarifRun
# ---------------------------------------------------------------------------


@dataclass
class SarifRun:
    """A single analysis run."""

    tool: SarifToolComponent = field(default_factory=SarifToolComponent)
    invocations: List[SarifInvocation] = field(default_factory=list)
    results: List[SarifResult] = field(default_factory=list)
    artifacts: List[SarifArtifact] = field(default_factory=list)
    logical_locations: List[SarifLogicalLocation] = field(default_factory=list)
    graphs: List[SarifGraph] = field(default_factory=list)
    taxonomies: List[SarifTaxonomy] = field(default_factory=list)
    column_kind: ColumnKind = ColumnKind.UTF16_CODE_UNITS
    language: str = "en-US"
    redaction_tokens: List[str] = field(default_factory=list)
    default_encoding: str = "utf-8"
    default_source_language: str = ""
    newline_sequences: List[str] = field(default_factory=lambda: ["\r\n", "\n"])
    baseline_guid: str = ""
    original_uri_base_ids: Dict[str, SarifArtifactLocation] = field(
        default_factory=dict
    )
    automation_details: Optional[Dict[str, Any]] = None
    conversion: Optional[Dict[str, Any]] = None
    external_property_file_references: Optional[Dict[str, Any]] = None
    thread_flow_locations: List[SarifThreadFlowLocation] = field(default_factory=list)
    web_requests: List[Dict[str, Any]] = field(default_factory=list)
    web_responses: List[Dict[str, Any]] = field(default_factory=list)
    special_locations: Optional[Dict[str, Any]] = None
    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    _artifact_uri_index: Dict[str, int] = field(
        default_factory=dict, repr=False
    )

    def add_result(self, result: SarifResult) -> int:
        idx = len(self.results)
        rule_index = self.tool.get_rule_index(result.rule_id)
        if rule_index >= 0:
            result.rule_index = rule_index
        self.results.append(result)
        return idx

    def add_artifact(self, artifact: SarifArtifact) -> int:
        uri = artifact.location.uri
        if uri in self._artifact_uri_index:
            return self._artifact_uri_index[uri]
        idx = len(self.artifacts)
        artifact.location.index = idx
        self.artifacts.append(artifact)
        self._artifact_uri_index[uri] = idx
        return idx

    def add_notification(self, notification: SarifNotification) -> None:
        if self.invocations:
            self.invocations[-1].add_notification(notification)
        else:
            inv = SarifInvocation()
            inv.add_notification(notification)
            self.invocations.append(inv)

    def add_graph(self, graph: SarifGraph) -> int:
        idx = len(self.graphs)
        self.graphs.append(graph)
        return idx

    def get_artifact_index(self, uri: str) -> int:
        return self._artifact_uri_index.get(uri, -1)

    def set_original_uri_base(self, base_id: str, uri: str) -> None:
        self.original_uri_base_ids[base_id] = SarifArtifactLocation(uri=uri)

    def get_results_by_rule(self, rule_id: str) -> List[SarifResult]:
        return [r for r in self.results if r.rule_id == rule_id]

    def get_results_by_level(self, level: SarifLevel) -> List[SarifResult]:
        return [r for r in self.results if r.level == level]

    def get_results_by_file(self, uri: str) -> List[SarifResult]:
        return [r for r in self.results if r.get_primary_uri() == uri]

    def count_by_level(self) -> Dict[str, int]:
        counts: Dict[str, int] = collections.Counter()
        for r in self.results:
            counts[r.level.value] += 1
        return dict(counts)

    def count_by_rule(self) -> Dict[str, int]:
        counts: Dict[str, int] = collections.Counter()
        for r in self.results:
            counts[r.rule_id] += 1
        return dict(counts)

    def to_sarif(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {"tool": {"driver": self.tool.to_sarif()}}
        if self.invocations:
            obj["invocations"] = [inv.to_sarif() for inv in self.invocations]
        obj["results"] = [r.to_sarif() for r in self.results]
        if self.artifacts:
            obj["artifacts"] = [a.to_sarif() for a in self.artifacts]
        if self.logical_locations:
            obj["logicalLocations"] = [
                ll.to_sarif() for ll in self.logical_locations
            ]
        if self.graphs:
            obj["graphs"] = [g.to_sarif() for g in self.graphs]
        if self.taxonomies:
            obj["taxonomies"] = [t.to_sarif() for t in self.taxonomies]
        obj["columnKind"] = self.column_kind.value
        if self.language:
            obj["language"] = self.language
        if self.default_encoding:
            obj["defaultEncoding"] = self.default_encoding
        if self.default_source_language:
            obj["defaultSourceLanguage"] = self.default_source_language
        if self.newline_sequences:
            obj["newlineSequences"] = list(self.newline_sequences)
        if self.baseline_guid:
            obj["baselineGuid"] = self.baseline_guid
        if self.original_uri_base_ids:
            obj["originalUriBaseIds"] = {
                k: v.to_sarif() for k, v in self.original_uri_base_ids.items()
            }
        if self.automation_details is not None:
            obj["automationDetails"] = self.automation_details
        if self.redaction_tokens:
            obj["redactionTokens"] = list(self.redaction_tokens)
        if self.properties:
            obj["properties"] = self.properties.to_sarif()
        return obj


# ---------------------------------------------------------------------------
# 19. SarifConversion
# ---------------------------------------------------------------------------


@dataclass
class SarifConversion:
    """Converts between SARIF and other formats."""

    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def sarif_to_simplified_json(self, sarif_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        simplified: List[Dict[str, Any]] = []
        for run in sarif_data.get("runs", []):
            tool_name = run.get("tool", {}).get("driver", {}).get("name", "")
            rules_list = run.get("tool", {}).get("driver", {}).get("rules", [])
            rules_map: Dict[str, Dict[str, Any]] = {}
            for rule in rules_list:
                rules_map[rule.get("id", "")] = rule

            for result in run.get("results", []):
                rule_id = result.get("ruleId", "")
                rule_info = rules_map.get(rule_id, {})
                locs = result.get("locations", [])
                primary_loc: Dict[str, Any] = {}
                if locs:
                    phys = locs[0].get("physicalLocation", {})
                    art = phys.get("artifactLocation", {})
                    region = phys.get("region", {})
                    primary_loc = {
                        "file": art.get("uri", ""),
                        "startLine": region.get("startLine", 0),
                        "startColumn": region.get("startColumn", 1),
                        "endLine": region.get("endLine", region.get("startLine", 0)),
                        "endColumn": region.get("endColumn"),
                    }

                entry: Dict[str, Any] = {
                    "tool": tool_name,
                    "ruleId": rule_id,
                    "ruleName": rule_info.get("name", ""),
                    "level": result.get("level", "warning"),
                    "message": result.get("message", {}).get("text", ""),
                    "location": primary_loc,
                }
                simplified.append(entry)
        return simplified

    def simplified_json_to_sarif(
        self,
        items: List[Dict[str, Any]],
        tool_name: str = "external-tool",
    ) -> Dict[str, Any]:
        rules_seen: Dict[str, int] = {}
        rules: List[Dict[str, Any]] = []
        results: List[Dict[str, Any]] = []

        for item in items:
            rule_id = item.get("ruleId", "UNKNOWN")
            if rule_id not in rules_seen:
                rules_seen[rule_id] = len(rules)
                rules.append({
                    "id": rule_id,
                    "name": item.get("ruleName", rule_id),
                    "shortDescription": {"text": item.get("ruleName", rule_id)},
                })

            loc = item.get("location", {})
            result: Dict[str, Any] = {
                "ruleId": rule_id,
                "ruleIndex": rules_seen[rule_id],
                "level": item.get("level", "warning"),
                "message": {"text": item.get("message", "")},
            }
            if loc:
                region: Dict[str, Any] = {}
                if "startLine" in loc:
                    region["startLine"] = loc["startLine"]
                if "startColumn" in loc:
                    region["startColumn"] = loc["startColumn"]
                if "endLine" in loc:
                    region["endLine"] = loc["endLine"]
                if "endColumn" in loc:
                    region["endColumn"] = loc["endColumn"]
                result["locations"] = [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": loc.get("file", "")},
                            "region": region,
                        }
                    }
                ]
            results.append(result)

        return {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": tool_name,
                            "rules": rules,
                        }
                    },
                    "results": results,
                }
            ],
        }

    def sarif_to_csv(self, sarif_data: Dict[str, Any]) -> str:
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "ruleId", "level", "message", "file", "startLine",
            "startColumn", "endLine", "endColumn",
        ])
        items = self.sarif_to_simplified_json(sarif_data)
        for item in items:
            loc = item.get("location", {})
            writer.writerow([
                item.get("ruleId", ""),
                item.get("level", ""),
                item.get("message", ""),
                loc.get("file", ""),
                loc.get("startLine", ""),
                loc.get("startColumn", ""),
                loc.get("endLine", ""),
                loc.get("endColumn", ""),
            ])
        return buf.getvalue()

    def sarif_to_junit_xml(self, sarif_data: Dict[str, Any]) -> str:
        items = self.sarif_to_simplified_json(sarif_data)
        files: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for item in items:
            file_key = item.get("location", {}).get("file", "unknown")
            files[file_key].append(item)

        lines: List[str] = ['<?xml version="1.0" encoding="UTF-8"?>']
        total_tests = len(items)
        total_failures = sum(
            1 for i in items if i.get("level") in ("error", "warning")
        )
        lines.append(
            f'<testsuites tests="{total_tests}" failures="{total_failures}">'
        )

        for file_path, file_items in sorted(files.items()):
            file_failures = sum(
                1 for i in file_items if i.get("level") in ("error", "warning")
            )
            suite_name = _xml_escape(file_path)
            lines.append(
                f'  <testsuite name="{suite_name}" tests="{len(file_items)}" '
                f'failures="{file_failures}">'
            )
            for item in file_items:
                rule_id = _xml_escape(item.get("ruleId", ""))
                level = item.get("level", "warning")
                msg = _xml_escape(item.get("message", ""))
                loc = item.get("location", {})
                start_line = loc.get("startLine", 0)
                test_name = f"{rule_id} at line {start_line}"
                lines.append(f'    <testcase name="{_xml_escape(test_name)}">')
                if level in ("error", "warning"):
                    tag = "failure" if level == "error" else "failure"
                    lines.append(
                        f'      <{tag} type="{rule_id}" message="{msg}">'
                    )
                    lines.append(f"        {msg}")
                    lines.append(f"      </{tag}>")
                lines.append("    </testcase>")
            lines.append("  </testsuite>")
        lines.append("</testsuites>")
        return "\n".join(lines)

    def sarif_to_text_summary(self, sarif_data: Dict[str, Any]) -> str:
        items = self.sarif_to_simplified_json(sarif_data)
        if not items:
            return "No issues found.\n"

        buf = io.StringIO()
        level_counts: Dict[str, int] = collections.Counter()
        rule_counts: Dict[str, int] = collections.Counter()
        for item in items:
            level_counts[item.get("level", "warning")] += 1
            rule_counts[item.get("ruleId", "UNKNOWN")] += 1

        buf.write("=" * 60 + "\n")
        buf.write("Refinement Type Analysis Summary\n")
        buf.write("=" * 60 + "\n\n")
        buf.write(f"Total issues: {len(items)}\n\n")
        buf.write("By severity:\n")
        for level in ["error", "warning", "note", "none"]:
            count = level_counts.get(level, 0)
            if count > 0:
                buf.write(f"  {level:>8}: {count}\n")
        buf.write("\nBy rule:\n")
        for rule_id, count in sorted(rule_counts.items()):
            buf.write(f"  {rule_id:>10}: {count}\n")
        buf.write("\n" + "-" * 60 + "\n")
        buf.write("Details:\n\n")

        for idx, item in enumerate(items, 1):
            loc = item.get("location", {})
            file_path = loc.get("file", "?")
            line = loc.get("startLine", "?")
            level = item.get("level", "warning")
            rule_id = item.get("ruleId", "?")
            msg = item.get("message", "")
            buf.write(f"  {idx}. [{level.upper()}] {rule_id}\n")
            buf.write(f"     {file_path}:{line}\n")
            buf.write(f"     {msg}\n\n")

        buf.write("=" * 60 + "\n")
        return buf.getvalue()


def _xml_escape(s: str) -> str:
    s = s.replace("&", "&amp;")
    s = s.replace("<", "&lt;")
    s = s.replace(">", "&gt;")
    s = s.replace('"', "&quot;")
    s = s.replace("'", "&apos;")
    return s


# ---------------------------------------------------------------------------
# 20. SarifValidator
# ---------------------------------------------------------------------------


@dataclass
class SarifValidationError:
    """A validation error found in SARIF data."""

    path: str = ""
    message: str = ""
    severity: str = "error"

    def __str__(self) -> str:
        return f"[{self.severity}] {self.path}: {self.message}"


@dataclass
class SarifValidator:
    """Validates SARIF output structure."""

    errors: List[SarifValidationError] = field(default_factory=list)
    warnings: List[SarifValidationError] = field(default_factory=list)

    def validate(self, sarif_data: Dict[str, Any]) -> bool:
        self.errors.clear()
        self.warnings.clear()
        self._validate_top_level(sarif_data)
        for run_idx, run in enumerate(sarif_data.get("runs", [])):
            self._validate_run(run, f"runs[{run_idx}]")
        return len(self.errors) == 0

    def _add_error(self, path: str, message: str) -> None:
        self.errors.append(SarifValidationError(path, message, "error"))

    def _add_warning(self, path: str, message: str) -> None:
        self.warnings.append(SarifValidationError(path, message, "warning"))

    def _validate_top_level(self, data: Dict[str, Any]) -> None:
        version = data.get("version")
        if version is None:
            self._add_error("version", "Missing required field 'version'")
        elif version != SARIF_VERSION:
            self._add_warning(
                "version",
                f"Expected version '{SARIF_VERSION}', got '{version}'",
            )

        schema = data.get("$schema")
        if schema is None:
            self._add_warning("$schema", "Missing '$schema' field")

        runs = data.get("runs")
        if runs is None:
            self._add_error("runs", "Missing required field 'runs'")
        elif not isinstance(runs, list):
            self._add_error("runs", "'runs' must be an array")
        elif len(runs) == 0:
            self._add_warning("runs", "'runs' array is empty")

    def _validate_run(self, run: Dict[str, Any], path: str) -> None:
        tool = run.get("tool")
        if tool is None:
            self._add_error(f"{path}.tool", "Missing required field 'tool'")
            return

        driver = tool.get("driver")
        if driver is None:
            self._add_error(
                f"{path}.tool.driver", "Missing required field 'driver'"
            )
            return

        if "name" not in driver:
            self._add_error(
                f"{path}.tool.driver.name",
                "Missing required field 'name' in driver",
            )

        rules = driver.get("rules", [])
        rule_ids: Set[str] = set()
        for rule_idx, rule in enumerate(rules):
            rule_path = f"{path}.tool.driver.rules[{rule_idx}]"
            self._validate_rule(rule, rule_path)
            rid = rule.get("id", "")
            if rid in rule_ids:
                self._add_warning(rule_path, f"Duplicate rule id '{rid}'")
            rule_ids.add(rid)

        results = run.get("results", [])
        num_rules = len(rules)
        for result_idx, result in enumerate(results):
            result_path = f"{path}.results[{result_idx}]"
            self._validate_result(result, result_path, rule_ids, num_rules)

        artifacts = run.get("artifacts", [])
        for art_idx, art in enumerate(artifacts):
            art_path = f"{path}.artifacts[{art_idx}]"
            self._validate_artifact(art, art_path)

    def _validate_rule(self, rule: Dict[str, Any], path: str) -> None:
        if "id" not in rule:
            self._add_error(path, "Missing required field 'id' in rule")
        rule_id = rule.get("id", "")
        if rule_id and not re.match(r"^[A-Za-z0-9_\-/.]+$", rule_id):
            self._add_warning(path, f"Rule id '{rule_id}' contains unusual characters")

        config = rule.get("defaultConfiguration", {})
        level = config.get("level", "")
        if level and level not in ("error", "warning", "note", "none"):
            self._add_error(
                f"{path}.defaultConfiguration.level",
                f"Invalid level '{level}'",
            )

    def _validate_result(
        self,
        result: Dict[str, Any],
        path: str,
        rule_ids: Set[str],
        num_rules: int,
    ) -> None:
        if "message" not in result:
            self._add_error(path, "Missing required field 'message'")
        else:
            msg = result["message"]
            if not isinstance(msg, dict):
                self._add_error(f"{path}.message", "'message' must be an object")
            elif "text" not in msg and "id" not in msg:
                self._add_error(
                    f"{path}.message",
                    "Message must have 'text' or 'id'",
                )

        level = result.get("level", "warning")
        if level not in ("error", "warning", "note", "none"):
            self._add_error(f"{path}.level", f"Invalid level '{level}'")

        rule_id = result.get("ruleId", "")
        if rule_id and rule_ids and rule_id not in rule_ids:
            self._add_warning(
                f"{path}.ruleId",
                f"ruleId '{rule_id}' not found in rules",
            )

        rule_index = result.get("ruleIndex")
        if rule_index is not None:
            if not isinstance(rule_index, int) or rule_index < 0:
                self._add_error(
                    f"{path}.ruleIndex",
                    f"ruleIndex must be a non-negative integer, got {rule_index}",
                )
            elif rule_index >= num_rules:
                self._add_error(
                    f"{path}.ruleIndex",
                    f"ruleIndex {rule_index} out of range (only {num_rules} rules)",
                )

        locations = result.get("locations", [])
        for loc_idx, loc in enumerate(locations):
            self._validate_location(loc, f"{path}.locations[{loc_idx}]")

    def _validate_location(self, loc: Dict[str, Any], path: str) -> None:
        phys = loc.get("physicalLocation")
        if phys is not None:
            region = phys.get("region")
            if region is not None:
                start_line = region.get("startLine")
                if start_line is not None and (
                    not isinstance(start_line, int) or start_line < 1
                ):
                    self._add_error(
                        f"{path}.region.startLine",
                        f"startLine must be a positive integer, got {start_line}",
                    )
                end_line = region.get("endLine")
                if end_line is not None and start_line is not None:
                    if isinstance(end_line, int) and isinstance(start_line, int):
                        if end_line < start_line:
                            self._add_error(
                                f"{path}.region.endLine",
                                f"endLine ({end_line}) < startLine ({start_line})",
                            )

    def _validate_artifact(self, artifact: Dict[str, Any], path: str) -> None:
        loc = artifact.get("location")
        if loc is None:
            self._add_warning(path, "Artifact missing 'location'")
        else:
            uri = loc.get("uri", "")
            if not uri:
                self._add_warning(f"{path}.location", "Artifact location has empty URI")

    def _validate_version_string(self, version: str, path: str) -> bool:
        if not version:
            return True
        pattern = r"^\d+\.\d+\.\d+(-[\w.]+)?(\+[\w.]+)?$"
        if not re.match(pattern, version):
            self._add_warning(
                path,
                f"Version string '{version}' does not follow semantic versioning",
            )
            return False
        return True

    def get_error_messages(self) -> List[str]:
        return [str(e) for e in self.errors]

    def get_warning_messages(self) -> List[str]:
        return [str(w) for w in self.warnings]

    def summary(self) -> str:
        return (
            f"Validation: {len(self.errors)} error(s), "
            f"{len(self.warnings)} warning(s)"
        )


# ---------------------------------------------------------------------------
# 21. GithubAnnotationAdapter
# ---------------------------------------------------------------------------


@dataclass
class GithubAnnotation:
    """A single GitHub check annotation."""

    path: str = ""
    start_line: int = 1
    end_line: int = 1
    start_column: Optional[int] = None
    end_column: Optional[int] = None
    annotation_level: str = "warning"
    message: str = ""
    title: str = ""
    raw_details: str = ""

    def to_dict(self) -> Dict[str, Any]:
        obj: Dict[str, Any] = {
            "path": self.path,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "annotation_level": self.annotation_level,
            "message": self.message,
        }
        if self.start_column is not None and self.start_line == self.end_line:
            obj["start_column"] = self.start_column
        if self.end_column is not None and self.start_line == self.end_line:
            obj["end_column"] = self.end_column
        if self.title:
            obj["title"] = self.title
        if self.raw_details:
            obj["raw_details"] = self.raw_details
        return obj


@dataclass
class GithubAnnotationAdapter:
    """Converts SARIF results to GitHub check annotation format."""

    repo_root: str = ""
    _level_map: Dict[str, str] = field(
        default_factory=lambda: {
            "error": "failure",
            "warning": "warning",
            "note": "notice",
            "none": "notice",
        }
    )

    def _map_level(self, sarif_level: str) -> str:
        return self._level_map.get(sarif_level, "warning")

    def _resolve_path(self, uri: str) -> str:
        path = uri
        if path.startswith("file://"):
            path = path[7:]
        if self.repo_root:
            root = self.repo_root.rstrip("/") + "/"
            if path.startswith(root):
                path = path[len(root):]
            elif path.startswith("/" + root):
                path = path[len(root) + 1:]
        if path.startswith("./"):
            path = path[2:]
        return path

    def _format_message(self, result: SarifResult) -> str:
        msg = str(result.message)
        if result.code_flows:
            flow_parts: List[str] = []
            for cf in result.code_flows:
                for tf in cf.thread_flows:
                    for tfl in tf.locations:
                        if tfl.location and tfl.location.message:
                            step_msg = str(tfl.location.message)
                            uri = tfl.location.get_uri()
                            line = tfl.location.get_start_line()
                            flow_parts.append(f"  → {uri}:{line}: {step_msg}")
            if flow_parts:
                msg += "\n\nFlow:\n" + "\n".join(flow_parts)
        return msg

    def result_to_annotation(self, result: SarifResult) -> Optional[GithubAnnotation]:
        if not result.locations:
            return None
        loc = result.locations[0]
        uri = loc.get_uri()
        if not uri:
            return None

        path = self._resolve_path(uri)
        start_line = loc.get_start_line()
        end_line = start_line
        start_col: Optional[int] = None
        end_col: Optional[int] = None

        if loc.physical_location and loc.physical_location.region:
            region = loc.physical_location.region
            if region.end_line is not None:
                end_line = region.end_line
            if region.start_column > 1:
                start_col = region.start_column
            if region.end_column is not None:
                end_col = region.end_column

        annotation_level = self._map_level(result.level.value)
        message = self._format_message(result)
        title = f"{result.rule_id}"
        rule = get_rule_by_id(result.rule_id)
        if rule and rule.short_description:
            title = f"{result.rule_id}: {rule.short_description.text}"

        return GithubAnnotation(
            path=path,
            start_line=start_line,
            end_line=end_line,
            start_column=start_col,
            end_column=end_col,
            annotation_level=annotation_level,
            message=message,
            title=title,
        )

    def results_to_annotations(
        self, results: List[SarifResult]
    ) -> List[GithubAnnotation]:
        annotations: List[GithubAnnotation] = []
        for result in results:
            if result.is_suppressed():
                continue
            annotation = self.result_to_annotation(result)
            if annotation is not None:
                annotations.append(annotation)
        return annotations

    def batch_annotations(
        self,
        annotations: List[GithubAnnotation],
        max_per_batch: int = MAX_GITHUB_ANNOTATIONS_PER_CALL,
    ) -> List[List[GithubAnnotation]]:
        batches: List[List[GithubAnnotation]] = []
        for i in range(0, len(annotations), max_per_batch):
            batches.append(annotations[i : i + max_per_batch])
        return batches

    def to_check_run_output(
        self,
        results: List[SarifResult],
        title: str = "Refinement Type Analysis",
        summary: str = "",
    ) -> Dict[str, Any]:
        annotations = self.results_to_annotations(results)
        if not summary:
            error_count = sum(1 for r in results if r.level == SarifLevel.ERROR)
            warn_count = sum(1 for r in results if r.level == SarifLevel.WARNING)
            note_count = sum(1 for r in results if r.level == SarifLevel.NOTE)
            summary = (
                f"Found {len(results)} issue(s): "
                f"{error_count} error(s), {warn_count} warning(s), "
                f"{note_count} note(s)."
            )

        batches = self.batch_annotations(annotations)
        first_batch = batches[0] if batches else []

        output: Dict[str, Any] = {
            "title": title,
            "summary": summary,
            "annotations": [a.to_dict() for a in first_batch],
        }
        return output

    def remaining_batches(
        self, results: List[SarifResult]
    ) -> List[List[Dict[str, Any]]]:
        annotations = self.results_to_annotations(results)
        batches = self.batch_annotations(annotations)
        remaining: List[List[Dict[str, Any]]] = []
        for batch in batches[1:]:
            remaining.append([a.to_dict() for a in batch])
        return remaining


# ---------------------------------------------------------------------------
# 22. CodeQLCompatibility
# ---------------------------------------------------------------------------


@dataclass
class CodeQLCompatibility:
    """Ensure output matches CodeQL SARIF expectations."""

    properties: SarifPropertyBag = field(default_factory=SarifPropertyBag)

    def ensure_codeql_properties(self, sarif_data: Dict[str, Any]) -> Dict[str, Any]:
        data = copy.deepcopy(sarif_data)
        for run in data.get("runs", []):
            driver = run.get("tool", {}).get("driver", {})
            if "semanticVersion" not in driver:
                driver["semanticVersion"] = driver.get("version", "0.0.0")

            for rule in driver.get("rules", []):
                props = rule.setdefault("properties", {})
                if "id" not in props:
                    props["id"] = rule.get("id", "")
                if "name" not in props:
                    props["name"] = rule.get("name", "")
                if "kind" not in props:
                    props["kind"] = "problem"
                if "precision" not in props:
                    level = rule.get("defaultConfiguration", {}).get("level", "warning")
                    if level == "error":
                        props["precision"] = "high"
                    elif level == "warning":
                        props["precision"] = "medium"
                    else:
                        props["precision"] = "low"
                if "problem.severity" not in props:
                    level = rule.get("defaultConfiguration", {}).get("level", "warning")
                    props["problem.severity"] = level

            for result in run.get("results", []):
                if "partialFingerprints" not in result:
                    result["partialFingerprints"] = {}
                pfp = result["partialFingerprints"]
                if "primaryLocationLineHash" not in pfp:
                    locs = result.get("locations", [])
                    if locs:
                        region = (
                            locs[0]
                            .get("physicalLocation", {})
                            .get("region", {})
                        )
                        line = region.get("startLine", 0)
                        rule_id = result.get("ruleId", "")
                        hash_input = f"{rule_id}:{line}"
                        pfp["primaryLocationLineHash"] = hashlib.sha256(
                            hash_input.encode()
                        ).hexdigest()[:16]

            if "columnKind" not in run:
                run["columnKind"] = "utf16CodeUnits"

            run.setdefault("properties", {})
            run["properties"]["semmle.formatSpecifier"] = "sarifv2.1.0"

        return data

    def add_query_metadata(
        self,
        sarif_data: Dict[str, Any],
        query_id: str = "",
        query_language: str = "python",
        query_kind: str = "problem",
    ) -> Dict[str, Any]:
        data = copy.deepcopy(sarif_data)
        for run in data.get("runs", []):
            for rule in run.get("tool", {}).get("driver", {}).get("rules", []):
                props = rule.setdefault("properties", {})
                if query_id:
                    props["queryId"] = query_id
                props["queryLanguage"] = query_language
                props["queryKind"] = query_kind
        return data

    def add_database_reference(
        self,
        sarif_data: Dict[str, Any],
        database_uri: str,
        source_location_prefix: str = "",
    ) -> Dict[str, Any]:
        data = copy.deepcopy(sarif_data)
        for run in data.get("runs", []):
            props = run.setdefault("properties", {})
            props["semmle.sourceLocationPrefix"] = source_location_prefix
            invocations = run.get("invocations", [])
            if invocations:
                inv_props = invocations[0].setdefault("properties", {})
                inv_props["databaseUri"] = database_uri
            else:
                run["invocations"] = [
                    {
                        "executionSuccessful": True,
                        "properties": {"databaseUri": database_uri},
                    }
                ]
        return data

    def format_for_upload(self, sarif_data: Dict[str, Any]) -> str:
        data = self.ensure_codeql_properties(sarif_data)
        return json.dumps(data, indent=2, default=str)

    def validate_codeql_compatibility(
        self, sarif_data: Dict[str, Any]
    ) -> List[str]:
        issues: List[str] = []
        for run_idx, run in enumerate(sarif_data.get("runs", [])):
            driver = run.get("tool", {}).get("driver", {})
            if "semanticVersion" not in driver:
                issues.append(
                    f"runs[{run_idx}]: driver missing 'semanticVersion'"
                )
            for r_idx, rule in enumerate(driver.get("rules", [])):
                props = rule.get("properties", {})
                if "kind" not in props:
                    issues.append(
                        f"runs[{run_idx}].rules[{r_idx}]: missing 'kind' property"
                    )
                if "precision" not in props:
                    issues.append(
                        f"runs[{run_idx}].rules[{r_idx}]: missing 'precision' property"
                    )
            if "columnKind" not in run:
                issues.append(f"runs[{run_idx}]: missing 'columnKind'")
        return issues


# ---------------------------------------------------------------------------
# 23. SarifReporter — main orchestrator
# ---------------------------------------------------------------------------


@dataclass
class SarifReporterConfig:
    """Configuration for the SARIF reporter."""

    tool_name: str = TOOL_NAME
    tool_version: str = TOOL_VERSION
    tool_info_uri: str = TOOL_INFO_URI
    tool_organization: str = TOOL_ORG
    include_artifacts: bool = True
    include_invocation: bool = True
    include_taxonomy: bool = True
    include_fingerprints: bool = True
    repo_root: str = ""
    severity_overrides: Dict[str, SarifLevel] = field(default_factory=dict)
    codeql_compatible: bool = False
    baseline_path: str = ""
    max_code_flow_steps: int = 100
    column_kind: ColumnKind = ColumnKind.UTF16_CODE_UNITS
    redaction_tokens: List[str] = field(default_factory=list)


@dataclass
class SarifReporter:
    """Main orchestrator for SARIF report generation."""

    config: SarifReporterConfig = field(default_factory=SarifReporterConfig)
    _run: SarifRun = field(default_factory=SarifRun, repr=False)
    _invocation: SarifInvocation = field(
        default_factory=SarifInvocation, repr=False
    )
    _baseline: SarifBaseline = field(
        default_factory=SarifBaseline, repr=False
    )
    _started: bool = field(default=False, repr=False)
    _finished: bool = field(default=False, repr=False)

    def __post_init__(self) -> None:
        self._setup_tool()

    def _setup_tool(self) -> None:
        self._run.tool = SarifToolComponent(
            name=self.config.tool_name,
            version=self.config.tool_version,
            semantic_version=self.config.tool_version,
            information_uri=self.config.tool_info_uri,
            organization=self.config.tool_organization,
            short_description=SarifMessage.from_text(
                "Refinement type inference and analysis for dynamic languages"
            ),
            full_description=SarifMessage.from_text(
                "Static analysis tool that infers refinement types for "
                "dynamic language programs and detects potential bugs "
                "such as null dereferences, array out-of-bounds access, "
                "division by zero, and type confusion."
            ),
            rules=list(BUILTIN_RULES),
        )
        self._run.column_kind = self.config.column_kind
        self._run.redaction_tokens = list(self.config.redaction_tokens)

        if self.config.include_taxonomy:
            taxonomy = SarifTaxonomy()
            self._run.taxonomies.append(taxonomy)
            self._run.tool.supported_taxonomies.append(
                {"name": taxonomy.name, "guid": taxonomy.guid}
            )

    def start(self, command_line: str = "", arguments: Optional[List[str]] = None) -> None:
        self._started = True
        self._invocation = SarifInvocation(
            command_line=command_line,
            arguments=arguments or [],
        )
        self._invocation.start()
        if self.config.repo_root:
            self._invocation.set_working_directory(self.config.repo_root)
            self._run.set_original_uri_base("SRCROOT", self.config.repo_root)

        if self.config.baseline_path:
            self._baseline.load_baseline(self.config.baseline_path)

    def finish(self, success: bool = True, exit_code: int = 0) -> None:
        self._finished = True
        self._invocation.finish(success=success, exit_code=exit_code)
        if self.config.include_invocation:
            self._run.invocations.append(self._invocation)

        if self.config.baseline_path and self._baseline.baseline_fingerprints:
            self._baseline.apply_baseline(self._run.results)

    def _resolve_level(self, rule_id: str, default_level: SarifLevel) -> SarifLevel:
        if rule_id in self.config.severity_overrides:
            return self.config.severity_overrides[rule_id]
        return default_level

    def _ensure_artifact(self, uri: str) -> int:
        existing = self._run.get_artifact_index(uri)
        if existing >= 0:
            return existing
        if self.config.include_artifacts:
            artifact = SarifArtifact(
                location=SarifArtifactLocation(uri=uri),
                roles=[ArtifactRole.ANALYSIS_TARGET],
            )
            artifact.guess_mime_type()
            artifact.guess_source_language()
            return self._run.add_artifact(artifact)
        return -1

    def add_bug_result(
        self,
        rule_id: str,
        message: str,
        file_uri: str,
        start_line: int,
        start_column: int = 1,
        end_line: Optional[int] = None,
        end_column: Optional[int] = None,
        function_name: str = "",
        code_context: str = "",
        flow_steps: Optional[List[Tuple[str, int, int, str]]] = None,
        fix_text: Optional[str] = None,
        fix_description: str = "Apply suggested fix",
        related_locations: Optional[
            List[Tuple[str, int, int, str]]
        ] = None,
        rank: float = -1.0,
        extra_properties: Optional[Dict[str, Any]] = None,
    ) -> SarifResult:
        rule = get_rule_by_id(rule_id)
        if rule is None:
            rule_level = SarifLevel.WARNING
        else:
            rule_level = rule.default_configuration.level

        level = self._resolve_level(rule_id, rule_level)

        self._ensure_artifact(file_uri)

        loc = SarifLocation.from_file_line(
            file_uri, start_line, start_column, end_line, end_column
        )
        if function_name:
            loc.logical_locations.append(
                SarifLogicalLocation(
                    name=function_name,
                    fully_qualified_name=function_name,
                    kind=LogicalLocationKind.FUNCTION,
                )
            )

        result = SarifResult(
            rule_id=rule_id,
            level=level,
            message=SarifMessage.from_text(message),
            locations=[loc],
        )

        if rank >= 0:
            result.rank = rank
        elif rule:
            result.rank = rule.default_configuration.rank

        if related_locations:
            for rel_uri, rel_line, rel_col, rel_msg in related_locations:
                rel_loc = SarifLocation.from_file_line(
                    rel_uri, rel_line, rel_col, message=rel_msg
                )
                result.add_related_location(rel_loc)
                self._ensure_artifact(rel_uri)

        if flow_steps:
            steps = flow_steps[: self.config.max_code_flow_steps]
            cf = SarifCodeFlow.from_path(message, steps)
            result.add_code_flow(cf)
            for step_uri, _, _, _ in steps:
                self._ensure_artifact(step_uri)

        if fix_text is not None and end_line is not None and end_column is not None:
            fix = SarifFix.simple_replacement(
                file_uri,
                start_line,
                start_column,
                end_line,
                end_column,
                fix_text,
                fix_description,
            )
            result.add_fix(fix)

        if self.config.include_fingerprints:
            result.compute_fingerprints(
                function_name=function_name,
                code_context=code_context,
                line_offset=start_line,
            )

        if extra_properties:
            for k, v in extra_properties.items():
                result.properties.set(k, v)

        cwe_id = _RULE_TO_CWE.get(rule_id)
        if cwe_id and self._run.taxonomies:
            taxonomy = self._run.taxonomies[0]
            taxon_idx = taxonomy.get_taxon_index(cwe_id)
            if taxon_idx >= 0:
                result.taxa.append({
                    "id": cwe_id,
                    "index": taxon_idx,
                    "toolComponent": {"index": 0},
                })

        self._run.add_result(result)
        return result

    def add_type_inference_result(
        self,
        file_uri: str,
        start_line: int,
        start_column: int,
        inferred_type: str,
        expected_type: str = "",
        variable_name: str = "",
        function_name: str = "",
        is_error: bool = False,
        confidence: float = 0.0,
    ) -> SarifResult:
        if is_error and expected_type:
            rule_id = "RT004"
            msg = (
                f"Type mismatch for '{variable_name}': "
                f"inferred type '{inferred_type}' is incompatible "
                f"with expected type '{expected_type}'"
            )
        elif is_error:
            rule_id = "RT004"
            msg = (
                f"Unable to verify type safety for '{variable_name}': "
                f"inferred type '{inferred_type}'"
            )
        else:
            rule_id = "RT006"
            msg = (
                f"Type inference result for '{variable_name}': "
                f"inferred as '{inferred_type}'"
            )

        props: Dict[str, Any] = {
            "inferredType": inferred_type,
            "variableName": variable_name,
        }
        if expected_type:
            props["expectedType"] = expected_type
        if confidence > 0:
            props["confidence"] = confidence

        return self.add_bug_result(
            rule_id=rule_id,
            message=msg,
            file_uri=file_uri,
            start_line=start_line,
            start_column=start_column,
            function_name=function_name,
            rank=confidence * 100 if confidence > 0 else -1.0,
            extra_properties=props,
        )

    def create_report(self, analysis_results: List[Dict[str, Any]]) -> None:
        for item in analysis_results:
            rule_name = item.get("rule", item.get("ruleId", ""))
            rule_id = RULE_NAME_TO_ID.get(rule_name, rule_name)
            if not get_rule_by_id(rule_id):
                rule_id = "RT004"

            message = item.get("message", "Issue detected")
            file_uri = item.get("file", item.get("uri", ""))
            start_line = item.get("line", item.get("startLine", 1))
            start_column = item.get("column", item.get("startColumn", 1))
            end_line = item.get("endLine")
            end_column = item.get("endColumn")
            function_name = item.get("function", "")
            code_context = item.get("context", "")

            flow_steps: Optional[List[Tuple[str, int, int, str]]] = None
            raw_steps = item.get("flowSteps", item.get("path", []))
            if raw_steps:
                flow_steps = []
                for step in raw_steps:
                    step_uri = step.get("file", file_uri)
                    step_line = step.get("line", 1)
                    step_col = step.get("column", 1)
                    step_msg = step.get("message", "")
                    flow_steps.append((step_uri, step_line, step_col, step_msg))

            fix_text = item.get("fixText")
            fix_desc = item.get("fixDescription", "Apply suggested fix")

            related: Optional[List[Tuple[str, int, int, str]]] = None
            raw_related = item.get("relatedLocations", [])
            if raw_related:
                related = []
                for rel in raw_related:
                    rel_uri = rel.get("file", "")
                    rel_line = rel.get("line", 1)
                    rel_col = rel.get("column", 1)
                    rel_msg = rel.get("message", "")
                    related.append((rel_uri, rel_line, rel_col, rel_msg))

            rank = item.get("confidence", item.get("rank", -1.0))
            extra_props = item.get("properties", {})

            self.add_bug_result(
                rule_id=rule_id,
                message=message,
                file_uri=file_uri,
                start_line=start_line,
                start_column=start_column,
                end_line=end_line,
                end_column=end_column,
                function_name=function_name,
                code_context=code_context,
                flow_steps=flow_steps,
                fix_text=fix_text,
                fix_description=fix_desc,
                related_locations=related,
                rank=rank,
                extra_properties=extra_props,
            )

    def generate(self) -> Dict[str, Any]:
        sarif: Dict[str, Any] = {
            "$schema": SARIF_SCHEMA,
            "version": SARIF_VERSION,
            "runs": [self._run.to_sarif()],
        }
        if self.config.codeql_compatible:
            compat = CodeQLCompatibility()
            sarif = compat.ensure_codeql_properties(sarif)
        return sarif

    def write_to_file(self, file_path: str, indent: int = 2) -> bool:
        try:
            sarif = self.generate()
            parent = os.path.dirname(file_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(sarif, f, indent=indent, default=str)
            return True
        except OSError:
            return False

    def write_baseline(self, file_path: str) -> bool:
        sarif = self.generate()
        return self._baseline.save_baseline(sarif, file_path)

    def get_run(self) -> SarifRun:
        return self._run

    def get_result_count(self) -> int:
        return len(self._run.results)

    def get_results_by_level(self, level: SarifLevel) -> List[SarifResult]:
        return self._run.get_results_by_level(level)

    def get_summary_stats(self) -> Dict[str, Any]:
        level_counts = self._run.count_by_level()
        rule_counts = self._run.count_by_rule()
        return {
            "totalResults": len(self._run.results),
            "byLevel": level_counts,
            "byRule": rule_counts,
            "artifactCount": len(self._run.artifacts),
        }

    def to_text_summary(self) -> str:
        sarif = self.generate()
        conv = SarifConversion()
        return conv.sarif_to_text_summary(sarif)

    def to_csv(self) -> str:
        sarif = self.generate()
        conv = SarifConversion()
        return conv.sarif_to_csv(sarif)

    def to_junit_xml(self) -> str:
        sarif = self.generate()
        conv = SarifConversion()
        return conv.sarif_to_junit_xml(sarif)

    def to_github_annotations(self) -> Dict[str, Any]:
        adapter = GithubAnnotationAdapter(repo_root=self.config.repo_root)
        return adapter.to_check_run_output(self._run.results)

    def validate(self) -> Tuple[bool, List[str], List[str]]:
        sarif = self.generate()
        validator = SarifValidator()
        is_valid = validator.validate(sarif)
        return is_valid, validator.get_error_messages(), validator.get_warning_messages()

    def merge_run(self, other: SarifReporter) -> None:
        for result in other._run.results:
            self._run.add_result(copy.deepcopy(result))
        for artifact in other._run.artifacts:
            uri = artifact.location.uri
            if self._run.get_artifact_index(uri) < 0:
                self._run.add_artifact(copy.deepcopy(artifact))
        for notification in other._invocation.tool_execution_notifications:
            self._invocation.add_notification(copy.deepcopy(notification))

    def add_notification(
        self,
        descriptor_id: str,
        message: str,
        level: SarifLevel = SarifLevel.NOTE,
    ) -> None:
        notification = SarifNotification(
            descriptor_id=descriptor_id,
            message=SarifMessage.from_text(message),
            level=level,
        )
        self._invocation.add_notification(notification)

    def add_graph(self, graph: SarifGraph) -> None:
        self._run.add_graph(graph)

    def reset(self) -> None:
        self._run = SarifRun()
        self._invocation = SarifInvocation()
        self._baseline = SarifBaseline()
        self._started = False
        self._finished = False
        self._setup_tool()
