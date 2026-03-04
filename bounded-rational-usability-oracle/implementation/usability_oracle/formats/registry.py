"""Format detection and registration.

Provides :class:`FormatRegistry` — a singleton registry that maps
format identifiers to parser classes, file extensions, and MIME types.
Supports auto-detection of format from content or file path.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Type


# ---------------------------------------------------------------------------
# FormatInfo
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FormatInfo:
    """Metadata for a registered accessibility-tree format.

    Attributes
    ----------
    id : str
        Short stable identifier (e.g. ``"html-aria"``).
    name : str
        Human-readable name.
    version : str
        Format / spec version.
    parser_class : str
        Fully qualified class name of the parser.
    extensions : tuple[str, ...]
        Associated file extensions (e.g. ``(".html", ".htm")``).
    mime_types : tuple[str, ...]
        Associated MIME types.
    schema_path : str
        Path to the bundled JSON schema (if any).
    """

    id: str
    name: str
    version: str = ""
    parser_class: str = ""
    extensions: tuple[str, ...] = ()
    mime_types: tuple[str, ...] = ()
    schema_path: str = ""


# ---------------------------------------------------------------------------
# Registry entry (internal)
# ---------------------------------------------------------------------------

@dataclass
class _RegistryEntry:
    info: FormatInfo
    detector: Optional[Callable[[str], bool]] = None


# ═══════════════════════════════════════════════════════════════════════════
# FormatRegistry (singleton)
# ═══════════════════════════════════════════════════════════════════════════

class FormatRegistry:
    """Singleton registry for accessibility-tree format parsers.

    Usage::

        registry = FormatRegistry.instance()
        registry.register("my-format", "mymod.MyParser", (".myfmt",))
        info = registry.detect(content)
        parser = registry.get_parser("my-format")
    """

    _instance: Optional[FormatRegistry] = None

    def __init__(self) -> None:
        self._entries: Dict[str, _RegistryEntry] = {}
        self._ext_index: Dict[str, str] = {}  # extension → format_id
        self._mime_index: Dict[str, str] = {}  # mime_type → format_id
        self._register_builtins()

    @classmethod
    def instance(cls) -> FormatRegistry:
        """Return the singleton registry instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        cls._instance = None

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        format_id: str,
        parser_class: str,
        extensions: Sequence[str] = (),
        mime_types: Sequence[str] = (),
        name: str = "",
        version: str = "",
        schema_path: str = "",
        detector: Optional[Callable[[str], bool]] = None,
    ) -> None:
        """Register a format.

        Parameters
        ----------
        format_id : str
            Unique identifier.
        parser_class : str
            Fully qualified class name.
        extensions : Sequence[str]
            File extensions (with leading dot).
        mime_types : Sequence[str]
            MIME types.
        name : str
            Human-readable name (defaults to *format_id*).
        version : str
            Version string.
        schema_path : str
            Path to bundled schema.
        detector : callable, optional
            Content-sniffing function ``(content: str) -> bool``.
        """
        info = FormatInfo(
            id=format_id,
            name=name or format_id,
            version=version,
            parser_class=parser_class,
            extensions=tuple(extensions),
            mime_types=tuple(mime_types),
            schema_path=schema_path,
        )
        self._entries[format_id] = _RegistryEntry(info=info, detector=detector)

        for ext in extensions:
            self._ext_index[ext.lower()] = format_id
        for mime in mime_types:
            self._mime_index[mime.lower()] = format_id

    def unregister(self, format_id: str) -> None:
        """Remove a format from the registry."""
        entry = self._entries.pop(format_id, None)
        if entry is None:
            return
        for ext in entry.info.extensions:
            self._ext_index.pop(ext.lower(), None)
        for mime in entry.info.mime_types:
            self._mime_index.pop(mime.lower(), None)

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, content: str) -> Optional[FormatInfo]:
        """Auto-detect format from content.

        Tries each registered detector in priority order.  Falls back
        to structural heuristics for JSON/XML content.

        Parameters
        ----------
        content : str
            Raw content string.

        Returns
        -------
        FormatInfo or None
            Detected format info, or None if unrecognised.
        """
        # Try registered detectors first
        for fid, entry in self._entries.items():
            if entry.detector is not None:
                try:
                    if entry.detector(content):
                        return entry.info
                except Exception:
                    continue

        # Structural heuristics
        stripped = content.strip()

        # XML-based formats
        if stripped.startswith("<?xml") or stripped.startswith("<hierarchy"):
            return self._entries.get("android-xml", _RegistryEntry(info=FormatInfo(id="android-xml", name="Android XML"))).info

        # JSON-based formats
        if stripped.startswith("{") or stripped.startswith("["):
            return self._detect_json(stripped)

        # HTML
        if re.search(r"<html|<body|<div|<!DOCTYPE", stripped, re.IGNORECASE):
            return self._entries.get("html-aria", _RegistryEntry(info=FormatInfo(id="html-aria", name="HTML ARIA"))).info

        # YAML
        if re.match(r"^---\s*$", stripped, re.MULTILINE) or "taskspec:" in stripped[:200]:
            return self._entries.get("yaml-taskspec", _RegistryEntry(info=FormatInfo(id="yaml-taskspec", name="YAML TaskSpec"))).info

        return None

    def detect_from_file(self, path: str) -> Optional[FormatInfo]:
        """Detect format from file extension and content.

        First checks the extension; if that matches, returns the
        corresponding format.  Otherwise reads the file and delegates
        to :meth:`detect`.

        Parameters
        ----------
        path : str
            File path.

        Returns
        -------
        FormatInfo or None
        """
        _, ext = os.path.splitext(path)
        ext = ext.lower()

        # Extension-based lookup
        if ext in self._ext_index:
            fid = self._ext_index[ext]
            return self._entries[fid].info

        # Content-based fallback
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read(8192)
            return self.detect(content)
        except (OSError, UnicodeDecodeError):
            return None

    # ------------------------------------------------------------------
    # Parser retrieval
    # ------------------------------------------------------------------

    def get_parser(self, format_id: str) -> Any:
        """Instantiate and return the parser for *format_id*.

        Parameters
        ----------
        format_id : str

        Returns
        -------
        object
            Parser instance.

        Raises
        ------
        KeyError
            If *format_id* is not registered.
        ImportError
            If the parser class cannot be imported.
        """
        if format_id not in self._entries:
            raise KeyError(f"Unknown format: {format_id!r}")

        class_path = self._entries[format_id].info.parser_class
        if not class_path:
            raise ImportError(
                f"No parser class registered for format {format_id!r}."
            )

        module_path, class_name = class_path.rsplit(".", 1)
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        return cls()

    # ------------------------------------------------------------------
    # Enumeration
    # ------------------------------------------------------------------

    def list_formats(self) -> list[FormatInfo]:
        """Return info for all registered formats."""
        return [e.info for e in self._entries.values()]

    def get_format(self, format_id: str) -> Optional[FormatInfo]:
        """Get info for a specific format, or None."""
        entry = self._entries.get(format_id)
        return entry.info if entry else None

    # ------------------------------------------------------------------
    # Built-in registration
    # ------------------------------------------------------------------

    def _register_builtins(self) -> None:
        """Register all built-in accessibility-tree formats."""

        self.register(
            format_id="html-aria",
            parser_class="usability_oracle.formats.aria.ARIAParser",
            extensions=(".html", ".htm"),
            mime_types=("text/html", "application/xhtml+xml"),
            name="HTML ARIA",
            version="1.2",
            detector=_detect_html_aria,
        )

        self.register(
            format_id="android-xml",
            parser_class="usability_oracle.formats.android.AndroidParser",
            extensions=(".xml",),
            mime_types=("text/xml", "application/xml"),
            name="Android XML Dump",
            version="1.0",
            detector=_detect_android_xml,
        )

        self.register(
            format_id="android-json",
            parser_class="usability_oracle.formats.android.AndroidParser",
            extensions=(),
            mime_types=("application/json",),
            name="Android JSON",
            version="1.0",
            detector=_detect_android_json,
        )

        self.register(
            format_id="json-a11y",
            parser_class="usability_oracle.formats.chrome_devtools.ChromeDevToolsParser",
            extensions=(".json",),
            mime_types=("application/json",),
            name="JSON Accessibility Tree",
            version="1.0",
            detector=_detect_json_a11y,
        )

        self.register(
            format_id="yaml-taskspec",
            parser_class="",
            extensions=(".yaml", ".yml"),
            mime_types=("application/yaml", "text/yaml"),
            name="YAML Task Specification",
            version="1.0",
            detector=_detect_yaml_taskspec,
        )

        self.register(
            format_id="axe-core",
            parser_class="usability_oracle.formats.axe_core.AxeCoreParser",
            extensions=(),
            mime_types=("application/json",),
            name="axe-core Results",
            version="4.x",
            detector=_detect_axe_core,
        )

        self.register(
            format_id="ios-a11y",
            parser_class="usability_oracle.formats.ios.IOSParser",
            extensions=(),
            mime_types=("application/json",),
            name="iOS Accessibility",
            version="1.0",
            detector=_detect_ios,
        )

        self.register(
            format_id="windows-uia",
            parser_class="usability_oracle.formats.windows.WindowsUIAParser",
            extensions=(),
            mime_types=("application/json",),
            name="Windows UIA",
            version="1.0",
            detector=_detect_windows_uia,
        )

        self.register(
            format_id="playwright",
            parser_class="usability_oracle.formats.playwright.PlaywrightParser",
            extensions=(),
            mime_types=("application/json",),
            name="Playwright Snapshot",
            version="1.0",
            detector=_detect_playwright,
        )

        self.register(
            format_id="selenium",
            parser_class="usability_oracle.formats.selenium_webdriver.SeleniumParser",
            extensions=(),
            mime_types=("application/json",),
            name="Selenium WebDriver DOM",
            version="1.0",
            detector=_detect_selenium,
        )

        self.register(
            format_id="cypress",
            parser_class="usability_oracle.formats.cypress.CypressParser",
            extensions=(),
            mime_types=("application/json",),
            name="Cypress Accessibility Audit",
            version="1.0",
            detector=_detect_cypress,
        )

        self.register(
            format_id="puppeteer",
            parser_class="usability_oracle.formats.puppeteer.PuppeteerParser",
            extensions=(),
            mime_types=("application/json",),
            name="Puppeteer Snapshot",
            version="1.0",
            detector=_detect_puppeteer,
        )

        self.register(
            format_id="react-testing-library",
            parser_class="usability_oracle.formats.react_testing_library.ReactTestingLibraryParser",
            extensions=(),
            mime_types=("text/plain", "text/html"),
            name="React Testing Library",
            version="1.0",
            detector=_detect_react_testing_library,
        )

        self.register(
            format_id="storybook",
            parser_class="usability_oracle.formats.storybook.StorybookParser",
            extensions=(),
            mime_types=("application/json",),
            name="Storybook Addon A11y",
            version="1.0",
            detector=_detect_storybook,
        )

        self.register(
            format_id="pa11y",
            parser_class="usability_oracle.formats.pa11y.Pa11yParser",
            extensions=(),
            mime_types=("application/json",),
            name="pa11y",
            version="1.0",
            detector=_detect_pa11y,
        )

        self.register(
            format_id="testing-library-queries",
            parser_class="usability_oracle.formats.testing_library_queries.TestingLibraryQueriesParser",
            extensions=(),
            mime_types=("application/json",),
            name="Testing Library Queries",
            version="1.0",
            detector=_detect_testing_library_queries,
        )

    def _detect_json(self, content: str) -> Optional[FormatInfo]:
        """Detect JSON sub-format from parsed content."""
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return None

        if isinstance(data, dict):
            # axe-core
            if "violations" in data and "passes" in data:
                return self._entries.get("axe-core", _RegistryEntry(info=FormatInfo(id="axe-core", name="axe-core"))).info

            # Chrome DevTools / generic a11y JSON
            if "nodes" in data:
                return self._entries.get("json-a11y", _RegistryEntry(info=FormatInfo(id="json-a11y", name="JSON a11y"))).info

            # Android JSON
            if "className" in data:
                cls_name = data.get("className", "")
                if "android." in cls_name or "androidx." in cls_name:
                    return self._entries.get("android-json", _RegistryEntry(info=FormatInfo(id="android-json", name="Android JSON"))).info

            # iOS
            if "elementType" in data or "accessibilityTraits" in data:
                return self._entries.get("ios-a11y", _RegistryEntry(info=FormatInfo(id="ios-a11y", name="iOS a11y"))).info

            # Windows UIA
            if "ControlType" in data or "AutomationId" in data:
                return self._entries.get("windows-uia", _RegistryEntry(info=FormatInfo(id="windows-uia", name="Windows UIA"))).info

            # Generic JSON with role
            if "role" in data:
                return self._entries.get("json-a11y", _RegistryEntry(info=FormatInfo(id="json-a11y", name="JSON a11y"))).info

        return self._entries.get("json-a11y", _RegistryEntry(info=FormatInfo(id="json-a11y", name="JSON a11y"))).info


# ═══════════════════════════════════════════════════════════════════════════
# Content detectors
# ═══════════════════════════════════════════════════════════════════════════

def _detect_html_aria(content: str) -> bool:
    """Detect HTML content with ARIA attributes."""
    return bool(re.search(
        r"<html|<body|<div|<!DOCTYPE",
        content[:2000],
        re.IGNORECASE,
    ))


def _detect_android_xml(content: str) -> bool:
    """Detect Android uiautomator XML dump."""
    stripped = content.strip()
    return (
        stripped.startswith("<hierarchy")
        or ("<hierarchy" in stripped[:500] and "android" in stripped[:500])
    )


def _detect_android_json(content: str) -> bool:
    """Detect Android AccessibilityNodeInfo JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    cls_name = data.get("className", "")
    return "android." in cls_name or "androidx." in cls_name


def _detect_json_a11y(content: str) -> bool:
    """Detect generic JSON accessibility tree."""
    stripped = content.strip()
    if not (stripped.startswith("{") or stripped.startswith("[")):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    if isinstance(data, dict):
        return "nodes" in data or ("role" in data and "nodeId" not in data)
    return False


def _detect_yaml_taskspec(content: str) -> bool:
    """Detect YAML task specification."""
    return "taskspec:" in content[:500] or content.strip().startswith("---")


def _detect_axe_core(content: str) -> bool:
    """Detect axe-core JSON results."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return "violations" in data and "passes" in data


def _detect_ios(content: str) -> bool:
    """Detect iOS accessibility JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return "elementType" in data or "accessibilityTraits" in data


def _detect_windows_uia(content: str) -> bool:
    """Detect Windows UIA JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return "ControlType" in data or "AutomationId" in data


def _detect_playwright(content: str) -> bool:
    """Detect Playwright accessibility snapshot JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("role") in ("WebArea", "RootWebArea")
        and "name" in data
        and "children" in data
        and "tag" not in data
        and "nodeId" not in data
    )


def _detect_selenium(content: str) -> bool:
    """Detect Selenium WebDriver DOM extraction JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return isinstance(data, dict) and "tag" in data and "rect" in data


def _detect_cypress(content: str) -> bool:
    """Detect Cypress accessibility audit JSON (cypress-axe)."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and "testTitle" in data
        and "results" in data
        and isinstance(data["results"], dict)
        and ("violations" in data["results"] or "passes" in data["results"])
    )


def _detect_puppeteer(content: str) -> bool:
    """Detect Puppeteer accessibility snapshot JSON."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and data.get("role") == "RootWebArea"
        and "name" in data
        and "tag" not in data
        and "nodeId" not in data
    )


def _detect_react_testing_library(content: str) -> bool:
    """Detect React Testing Library logRoles() output."""
    return bool(re.search(
        r"^\w[\w-]*:\s*\n\s*\nName\s+\"",
        content[:2000],
        re.MULTILINE,
    ))


def _detect_storybook(content: str) -> bool:
    """Detect Storybook addon-a11y JSON output."""
    stripped = content.strip()
    if not stripped.startswith("{"):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    return (
        isinstance(data, dict)
        and "storyId" in data
        and "axeResults" in data
    )


def _detect_pa11y(content: str) -> bool:
    """Detect pa11y JSON output."""
    stripped = content.strip()
    if not stripped.startswith("["):
        return False
    try:
        data = json.loads(stripped[:8192])
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return isinstance(first, dict) and "code" in first and "selector" in first and "runner" in first


def _detect_testing_library_queries(content: str) -> bool:
    """Detect Testing Library structured query results JSON."""
    stripped = content.strip()
    if not stripped.startswith("["):
        return False
    try:
        data = json.loads(stripped[:4096])
    except (json.JSONDecodeError, ValueError):
        return False
    if not isinstance(data, list) or not data:
        return False
    first = data[0]
    return (
        isinstance(first, dict)
        and "role" in first
        and "name" in first
        and ("hidden" in first or "pressed" in first or "checked" in first)
    )
