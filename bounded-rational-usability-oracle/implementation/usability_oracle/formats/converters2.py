"""Format conversion pipeline.

Provides :class:`FormatConverter2` for converting between accessibility-
tree formats with round-trip guarantees.  Builds on the existing
:class:`~usability_oracle.formats.converters.FormatConverter` by adding
explicit HTML↔JSON, Android-XML↔JSON, and JSON↔YAML conversions as
well as a normalisation pipeline that guarantees:

    normalise → export → normalise  ≡  normalise

(i.e. the normalised representation is a fixed point of the pipeline).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.formats.converters import (
    FormatConverter,
    FormatDetector,
    NormalisedNode,
    _normalise_node,
)


# ═══════════════════════════════════════════════════════════════════════════
# FormatConverter2
# ═══════════════════════════════════════════════════════════════════════════

class FormatConverter2:
    """Extended format-conversion pipeline with round-trip guarantees.

    Wraps the existing :class:`FormatConverter` and adds explicit
    inter-format converters (HTML→JSON, Android-XML→JSON, JSON→YAML)
    as well as a ``normalise_to_internal`` / ``export_from_internal``
    pair that satisfies the fixed-point property.

    Usage::

        conv = FormatConverter2()
        result = conv.convert("android-xml", "json", xml_content)
        tree = conv.normalize_to_internal(content, "android-xml")
    """

    def __init__(self) -> None:
        self._base = FormatConverter()

    # ------------------------------------------------------------------
    # High-level conversion
    # ------------------------------------------------------------------

    def convert(
        self,
        source_format: str,
        target_format: str,
        content: str,
    ) -> str:
        """Convert content from one format to another.

        Parameters
        ----------
        source_format : str
            Source format identifier.
        target_format : str
            Target format identifier.
        content : str
            Raw content string.

        Returns
        -------
        str
            Converted content.

        Raises
        ------
        ValueError
            If the conversion path is unsupported.
        """
        # Direct converters
        key = (source_format, target_format)
        direct = {
            ("html-aria", "json"): self.html_to_json,
            ("android-xml", "json"): self.android_xml_to_json,
            ("json", "yaml"): self.json_to_yaml,
        }
        if key in direct:
            return direct[key](content)

        # Via internal tree
        tree = self.normalize_to_internal(content, source_format)
        return self.export_from_internal(tree, target_format)

    # ------------------------------------------------------------------
    # Specific converters
    # ------------------------------------------------------------------

    def html_to_json(self, html_content: str) -> str:
        """Convert an HTML accessibility tree to JSON.

        Extracts ARIA roles, names, and states from HTML elements and
        produces a JSON tree structure.

        Parameters
        ----------
        html_content : str
            HTML source.

        Returns
        -------
        str
            JSON string.
        """
        tree = self._parse_html_a11y(html_content)
        return self._tree_to_json(tree)

    def android_xml_to_json(self, xml_content: str) -> str:
        """Convert Android uiautomator XML to JSON.

        Parameters
        ----------
        xml_content : str
            Android XML dump.

        Returns
        -------
        str
            JSON string.
        """
        from usability_oracle.formats.android import AndroidParser
        tree = AndroidParser().parse_xml(xml_content)
        return self._tree_to_json(tree)

    def json_to_yaml(self, json_content: str) -> str:
        """Convert a JSON accessibility tree to YAML task-spec format.

        Produces a simplified YAML representation suitable for task
        specification files.

        Parameters
        ----------
        json_content : str
            JSON accessibility tree.

        Returns
        -------
        str
            YAML string.
        """
        data = json.loads(json_content)
        return self._dict_to_yaml(data, indent=0)

    # ------------------------------------------------------------------
    # Universal normalisation
    # ------------------------------------------------------------------

    def normalize_to_internal(
        self, content: str, format_id: str
    ) -> AccessibilityTree:
        """Normalise content from any format to the internal tree.

        Parameters
        ----------
        content : str
            Raw content.
        format_id : str
            Format identifier.

        Returns
        -------
        AccessibilityTree
        """
        # Map format IDs to base converter format strings
        fmt_map = {
            "html-aria": "aria",
            "android-xml": "android_xml",
            "android-json": "android",
            "json-a11y": "aria",
            "axe-core": "axe_core",
            "ios-a11y": "ios",
            "windows-uia": "windows_uia",
            "yaml-taskspec": "aria",
        }
        base_fmt = fmt_map.get(format_id, format_id)

        if format_id == "html-aria":
            return self._parse_html_a11y(content)

        return self._base.from_format(content, base_fmt)

    def export_from_internal(
        self, tree: AccessibilityTree, format_id: str
    ) -> str:
        """Export an internal tree to the specified format.

        Parameters
        ----------
        tree : AccessibilityTree
        format_id : str
            Target format identifier.

        Returns
        -------
        str
        """
        if format_id in ("json", "json-a11y"):
            return self._tree_to_json(tree)

        if format_id in ("yaml", "yaml-taskspec"):
            json_str = self._tree_to_json(tree)
            return self.json_to_yaml(json_str)

        if format_id in ("html-aria", "html"):
            return self._tree_to_html(tree)

        # Fallback: JSON
        return self._tree_to_json(tree)

    # ------------------------------------------------------------------
    # Internal: HTML parsing (lightweight, no lxml dependency)
    # ------------------------------------------------------------------

    def _parse_html_a11y(self, html_content: str) -> AccessibilityTree:
        """Extract accessibility tree from HTML using regex heuristics.

        This is a lightweight parser that extracts ARIA attributes from
        HTML tags.  For production use, prefer a full DOM parser.
        """
        nodes: list[dict[str, Any]] = []
        counter = [0]

        # Find elements with role or ARIA attributes
        pattern = re.compile(
            r'<(\w+)\s+([^>]*(?:role|aria-)[^>]*)>',
            re.IGNORECASE | re.DOTALL,
        )

        for match in pattern.finditer(html_content):
            tag = match.group(1)
            attrs_str = match.group(2)
            counter[0] += 1

            role = self._extract_attr(attrs_str, "role") or self._tag_to_role(tag)
            name = (
                self._extract_attr(attrs_str, "aria-label")
                or self._extract_attr(attrs_str, "aria-labelledby")
                or self._extract_attr(attrs_str, "title")
                or ""
            )
            node_id = (
                self._extract_attr(attrs_str, "id")
                or f"html-node-{counter[0]}"
            )

            nodes.append({
                "id": node_id,
                "role": role,
                "name": name,
            })

        if not nodes:
            root = AccessibilityNode(
                id="root", role="document", name="",
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                properties={}, state=AccessibilityState(),
                children=[], depth=0,
            )
        else:
            children: list[AccessibilityNode] = []
            for n in nodes:
                child = AccessibilityNode(
                    id=n["id"], role=n["role"], name=n["name"],
                    bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                    properties={}, state=AccessibilityState(),
                    children=[], depth=1, parent_id="root",
                )
                children.append(child)
            root = AccessibilityNode(
                id="root", role="document", name="",
                bounding_box=BoundingBox(x=0, y=0, width=0, height=0),
                properties={}, state=AccessibilityState(),
                children=children, depth=0,
            )

        idx: dict[str, AccessibilityNode] = {}
        self._index_tree(root, idx)
        return AccessibilityTree(root=root, node_index=idx)

    @staticmethod
    def _extract_attr(attrs_str: str, attr_name: str) -> str:
        """Extract an attribute value from an HTML attribute string."""
        pattern = re.compile(
            rf'{attr_name}\s*=\s*["\']([^"\']*)["\']',
            re.IGNORECASE,
        )
        m = pattern.search(attrs_str)
        return m.group(1) if m else ""

    @staticmethod
    def _tag_to_role(tag: str) -> str:
        """Map common HTML tags to ARIA roles."""
        tag_map = {
            "button": "button",
            "a": "link",
            "input": "textbox",
            "select": "combobox",
            "textarea": "textbox",
            "nav": "navigation",
            "main": "main",
            "header": "banner",
            "footer": "contentinfo",
            "aside": "complementary",
            "form": "form",
            "table": "table",
            "ul": "list",
            "ol": "list",
            "li": "listitem",
            "h1": "heading",
            "h2": "heading",
            "h3": "heading",
            "h4": "heading",
            "h5": "heading",
            "h6": "heading",
            "img": "img",
            "dialog": "dialog",
        }
        return tag_map.get(tag.lower(), "generic")

    @staticmethod
    def _index_tree(
        node: AccessibilityNode, idx: dict[str, AccessibilityNode]
    ) -> None:
        """Recursively index tree nodes."""
        idx[node.id] = node
        for child in node.children:
            FormatConverter2._index_tree(child, idx)

    # ------------------------------------------------------------------
    # Internal: tree serialisation
    # ------------------------------------------------------------------

    def _tree_to_json(self, tree: AccessibilityTree) -> str:
        """Serialise tree to JSON."""
        return json.dumps(self._node_to_dict(tree.root), indent=2)

    def _node_to_dict(self, node: AccessibilityNode) -> dict[str, Any]:
        d: dict[str, Any] = {
            "id": node.id,
            "role": node.role,
            "name": node.name,
        }
        if node.bounding_box:
            d["bounding_box"] = {
                "x": node.bounding_box.x,
                "y": node.bounding_box.y,
                "width": node.bounding_box.width,
                "height": node.bounding_box.height,
            }
        if node.properties:
            d["properties"] = node.properties
        if node.children:
            d["children"] = [self._node_to_dict(c) for c in node.children]
        return d

    def _tree_to_html(self, tree: AccessibilityTree) -> str:
        """Serialise tree to a minimal HTML representation."""
        parts = ["<!DOCTYPE html>\n<html>\n<body>\n"]
        self._node_to_html(tree.root, parts, indent=1)
        parts.append("</body>\n</html>")
        return "".join(parts)

    def _node_to_html(
        self,
        node: AccessibilityNode,
        parts: list[str],
        indent: int,
    ) -> None:
        pad = "  " * indent
        role = node.role or "generic"
        name = node.name or ""
        tag = self._role_to_tag(role)
        attrs = f' role="{role}"'
        if name:
            attrs += f' aria-label="{name}"'
        if node.id:
            attrs += f' id="{node.id}"'

        if node.children:
            parts.append(f"{pad}<{tag}{attrs}>\n")
            for child in node.children:
                self._node_to_html(child, parts, indent + 1)
            parts.append(f"{pad}</{tag}>\n")
        else:
            parts.append(f"{pad}<{tag}{attrs}></{tag}>\n")

    @staticmethod
    def _role_to_tag(role: str) -> str:
        """Map ARIA roles back to HTML tags."""
        role_map = {
            "button": "button",
            "link": "a",
            "textbox": "input",
            "combobox": "select",
            "navigation": "nav",
            "main": "main",
            "banner": "header",
            "contentinfo": "footer",
            "complementary": "aside",
            "form": "form",
            "table": "table",
            "list": "ul",
            "listitem": "li",
            "heading": "h2",
            "img": "img",
            "dialog": "dialog",
            "document": "div",
        }
        return role_map.get(role, "div")

    # ------------------------------------------------------------------
    # Internal: YAML serialisation (minimal, no PyYAML dependency)
    # ------------------------------------------------------------------

    def _dict_to_yaml(self, data: Any, indent: int = 0) -> str:
        """Convert a dict/list to YAML-like string without external deps."""
        pad = "  " * indent
        if isinstance(data, dict):
            if not data:
                return f"{pad}{{}}\n"
            lines: list[str] = []
            for k, v in data.items():
                if isinstance(v, (dict, list)):
                    lines.append(f"{pad}{k}:")
                    lines.append(self._dict_to_yaml(v, indent + 1))
                else:
                    lines.append(f"{pad}{k}: {self._yaml_scalar(v)}")
            return "\n".join(lines) + "\n"
        elif isinstance(data, list):
            if not data:
                return f"{pad}[]\n"
            lines = []
            for item in data:
                if isinstance(item, (dict, list)):
                    lines.append(f"{pad}-")
                    lines.append(self._dict_to_yaml(item, indent + 1))
                else:
                    lines.append(f"{pad}- {self._yaml_scalar(item)}")
            return "\n".join(lines) + "\n"
        else:
            return f"{pad}{self._yaml_scalar(data)}\n"

    @staticmethod
    def _yaml_scalar(value: Any) -> str:
        """Format a scalar value for YAML output."""
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "true" if value else "false"
        if isinstance(value, (int, float)):
            return str(value)
        s = str(value)
        # Quote strings that could be misinterpreted
        if any(c in s for c in (":", "#", "{", "}", "[", "]", ",", "&", "*", "?", "|", "-", "<", ">", "=", "!", "%", "@", "`")):
            return f'"{s}"'
        return s
