"""
usability_oracle.aria.taxonomy — Complete WAI-ARIA 1.2 role taxonomy.

Encodes all 82 roles from the WAI-ARIA 1.2 specification (W3C
Recommendation, 6 June 2023) with their superclass/subclass chains,
required/supported properties and states, ownership constraints, and
naming rules.

Reference: https://www.w3.org/TR/wai-aria-1.2/#role_definitions

Usage::

    from usability_oracle.aria.taxonomy import ROLE_TAXONOMY, get_role
    button = get_role("button")
    assert button.category == RoleCategory.WIDGET
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Sequence

from usability_oracle.aria.types import AriaRole, RoleCategory


def _r(
    name: str,
    category: RoleCategory,
    *,
    superclasses: FrozenSet[str] = frozenset(),
    subclasses: FrozenSet[str] = frozenset(),
    required_props: FrozenSet[str] = frozenset(),
    supported_props: FrozenSet[str] = frozenset(),
    required_states: FrozenSet[str] = frozenset(),
    supported_states: FrozenSet[str] = frozenset(),
    required_owned: FrozenSet[str] = frozenset(),
    required_context: FrozenSet[str] = frozenset(),
    allowed_children: FrozenSet[str] = frozenset(),
    is_abstract: bool = False,
    name_from: str = "author",
    is_presentational: bool = False,
    desc: str = "",
) -> AriaRole:
    """Shorthand constructor for taxonomy entries."""
    return AriaRole(
        name=name,
        category=category,
        superclass_roles=superclasses,
        subclass_roles=subclasses,
        required_properties=required_props,
        supported_properties=supported_props,
        required_states=required_states,
        supported_states=supported_states,
        required_owned_elements=required_owned,
        required_context_roles=required_context,
        allowed_child_roles=allowed_children,
        is_abstract=is_abstract,
        name_from=name_from,
        is_presentational=is_presentational,
        description=desc,
    )


_W = RoleCategory.WIDGET
_D = RoleCategory.DOCUMENT_STRUCTURE
_L = RoleCategory.LANDMARK
_LR = RoleCategory.LIVE_REGION
_WN = RoleCategory.WINDOW
_A = RoleCategory.ABSTRACT


# ═══════════════════════════════════════════════════════════════════════════
# Abstract roles (§5.3.1) — 12 roles
# ═══════════════════════════════════════════════════════════════════════════

_ABSTRACT_ROLES: Dict[str, AriaRole] = {
    "command": _r(
        "command", _A, is_abstract=True,
        superclasses=frozenset({"widget"}),
        subclasses=frozenset({"button", "link", "menuitem"}),
        desc="Base for interactive command elements.",
    ),
    "composite": _r(
        "composite", _A, is_abstract=True,
        superclasses=frozenset({"widget"}),
        subclasses=frozenset({"grid", "select", "spinbutton", "tablist"}),
        supported_states=frozenset({"activedescendant"}),
        desc="Widget that contains navigable descendants or owned children.",
    ),
    "input": _r(
        "input", _A, is_abstract=True,
        superclasses=frozenset({"widget"}),
        subclasses=frozenset({
            "checkbox", "combobox", "option", "radio",
            "slider", "spinbutton", "textbox",
        }),
        desc="Generic interactive control accepting user input.",
    ),
    "landmark": _r(
        "landmark", _A, is_abstract=True,
        superclasses=frozenset({"section"}),
        subclasses=frozenset({
            "banner", "complementary", "contentinfo", "form",
            "main", "navigation", "region", "search",
        }),
        desc="Navigational landmark region.",
    ),
    "range": _r(
        "range", _A, is_abstract=True,
        superclasses=frozenset({"widget"}),
        subclasses=frozenset({"meter", "progressbar", "scrollbar", "slider", "spinbutton"}),
        supported_props=frozenset({"valuemax", "valuemin", "valuenow", "valuetext"}),
        desc="Element representing a range of values.",
    ),
    "roletype": _r(
        "roletype", _A, is_abstract=True,
        subclasses=frozenset({"structure", "widget", "window"}),
        desc="Root of the ARIA role taxonomy.",
    ),
    "section": _r(
        "section", _A, is_abstract=True,
        superclasses=frozenset({"structure"}),
        subclasses=frozenset({
            "alert", "blockquote", "caption", "cell", "definition",
            "figure", "group", "img", "landmark", "list", "listitem",
            "log", "marquee", "math", "note", "status", "table",
            "tabpanel", "term", "tooltip",
        }),
        desc="Renderable structural containment for document content.",
    ),
    "sectionhead": _r(
        "sectionhead", _A, is_abstract=True,
        superclasses=frozenset({"structure"}),
        subclasses=frozenset({"columnheader", "heading", "rowheader", "tab"}),
        name_from="contents",
        desc="Label or summary for a section.",
    ),
    "select": _r(
        "select", _A, is_abstract=True,
        superclasses=frozenset({"composite", "group"}),
        subclasses=frozenset({"listbox", "menu", "radiogroup", "tablist", "tree"}),
        supported_props=frozenset({"orientation"}),
        desc="Composite widget offering a list of choices.",
    ),
    "structure": _r(
        "structure", _A, is_abstract=True,
        superclasses=frozenset({"roletype"}),
        subclasses=frozenset({
            "application", "document", "generic", "presentation",
            "rowgroup", "section", "sectionhead", "separator",
        }),
        desc="Structural element of the document.",
    ),
    "widget": _r(
        "widget", _A, is_abstract=True,
        superclasses=frozenset({"roletype"}),
        subclasses=frozenset({
            "command", "composite", "gridcell", "input", "progressbar",
            "range", "row", "scrollbar", "separator", "tab",
        }),
        desc="Interactive component of a GUI.",
    ),
    "window": _r(
        "window", _A, is_abstract=True,
        superclasses=frozenset({"roletype"}),
        subclasses=frozenset({"alertdialog", "dialog"}),
        supported_props=frozenset({"modal"}),
        desc="Browser or application window.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Widget roles (§5.3.2) — 29 roles
# ═══════════════════════════════════════════════════════════════════════════

_WIDGET_ROLES: Dict[str, AriaRole] = {
    "button": _r(
        "button", _W,
        superclasses=frozenset({"command"}),
        name_from="contents",
        supported_states=frozenset({"expanded", "pressed"}),
        desc="Clickable element that triggers an action.",
    ),
    "checkbox": _r(
        "checkbox", _W,
        superclasses=frozenset({"input"}),
        required_states=frozenset({"checked"}),
        name_from="contents",
        supported_states=frozenset({"readonly", "required"}),
        desc="Checkable interactive control.",
    ),
    "combobox": _r(
        "combobox", _W,
        superclasses=frozenset({"input"}),
        required_states=frozenset({"expanded"}),
        supported_props=frozenset({
            "autocomplete", "controls", "activedescendant", "readonly", "required",
        }),
        desc="Composite widget combining input with a popup.",
    ),
    "grid": _r(
        "grid", _W,
        superclasses=frozenset({"composite", "table"}),
        required_owned=frozenset({"row", "rowgroup"}),
        supported_props=frozenset({"multiselectable", "readonly"}),
        desc="Interactive grid or table widget.",
    ),
    "gridcell": _r(
        "gridcell", _W,
        superclasses=frozenset({"cell", "widget"}),
        required_context=frozenset({"row"}),
        supported_states=frozenset({"readonly", "required", "selected"}),
        name_from="contents",
        desc="Cell in a grid or treegrid.",
    ),
    "link": _r(
        "link", _W,
        superclasses=frozenset({"command"}),
        name_from="contents",
        supported_states=frozenset({"expanded"}),
        desc="Interactive reference to a resource.",
    ),
    "listbox": _r(
        "listbox", _W,
        superclasses=frozenset({"select"}),
        required_owned=frozenset({"group", "option"}),
        supported_props=frozenset({
            "multiselectable", "readonly", "required", "orientation",
        }),
        supported_states=frozenset({"expanded"}),
        desc="Widget allowing selection from a list of options.",
    ),
    "menu": _r(
        "menu", _W,
        superclasses=frozenset({"select"}),
        required_owned=frozenset({"group", "menuitem", "menuitemcheckbox", "menuitemradio"}),
        supported_props=frozenset({"orientation"}),
        desc="Widget offering a list of actions or functions.",
    ),
    "menubar": _r(
        "menubar", _W,
        superclasses=frozenset({"menu"}),
        required_owned=frozenset({"group", "menuitem", "menuitemcheckbox", "menuitemradio"}),
        supported_props=frozenset({"orientation"}),
        desc="Menu bar — usually horizontal, contains menu items.",
    ),
    "menuitem": _r(
        "menuitem", _W,
        superclasses=frozenset({"command"}),
        required_context=frozenset({"group", "menu", "menubar"}),
        name_from="contents",
        supported_props=frozenset({"posinset", "setsize"}),
        supported_states=frozenset({"expanded"}),
        desc="Option in a menu.",
    ),
    "menuitemcheckbox": _r(
        "menuitemcheckbox", _W,
        superclasses=frozenset({"checkbox", "menuitem"}),
        required_context=frozenset({"group", "menu", "menubar"}),
        required_states=frozenset({"checked"}),
        name_from="contents",
        desc="Checkable menu item.",
    ),
    "menuitemradio": _r(
        "menuitemradio", _W,
        superclasses=frozenset({"menuitemcheckbox", "radio"}),
        required_context=frozenset({"group", "menu", "menubar"}),
        required_states=frozenset({"checked"}),
        name_from="contents",
        desc="Radio-button menu item in a mutually exclusive set.",
    ),
    "option": _r(
        "option", _W,
        superclasses=frozenset({"input"}),
        required_context=frozenset({"group", "listbox"}),
        name_from="contents",
        supported_states=frozenset({"checked", "selected"}),
        supported_props=frozenset({"posinset", "setsize"}),
        desc="Selectable item in a listbox.",
    ),
    "progressbar": _r(
        "progressbar", _W,
        superclasses=frozenset({"range", "widget"}),
        supported_props=frozenset({"valuemax", "valuemin", "valuenow", "valuetext"}),
        desc="Element showing progress of a task.",
    ),
    "radio": _r(
        "radio", _W,
        superclasses=frozenset({"input"}),
        required_states=frozenset({"checked"}),
        name_from="contents",
        supported_props=frozenset({"posinset", "setsize"}),
        desc="Checkable input in a mutually exclusive group.",
    ),
    "radiogroup": _r(
        "radiogroup", _W,
        superclasses=frozenset({"select"}),
        required_owned=frozenset({"radio"}),
        supported_states=frozenset({"readonly", "required"}),
        supported_props=frozenset({"orientation"}),
        desc="Group of radio buttons.",
    ),
    "scrollbar": _r(
        "scrollbar", _W,
        superclasses=frozenset({"range", "widget"}),
        required_props=frozenset({"controls", "valuenow"}),
        supported_props=frozenset({"orientation", "valuemax", "valuemin", "valuetext"}),
        desc="Graphical scrollbar object.",
    ),
    "searchbox": _r(
        "searchbox", _W,
        superclasses=frozenset({"textbox"}),
        desc="Textbox intended for specifying search criteria.",
    ),
    "slider": _r(
        "slider", _W,
        superclasses=frozenset({"input", "range"}),
        required_props=frozenset({"valuenow"}),
        supported_props=frozenset({
            "orientation", "valuemax", "valuemin", "valuetext", "readonly",
        }),
        desc="Input where the user selects from a range.",
    ),
    "spinbutton": _r(
        "spinbutton", _W,
        superclasses=frozenset({"composite", "input", "range"}),
        supported_props=frozenset({
            "valuemax", "valuemin", "valuenow", "valuetext",
            "readonly", "required",
        }),
        desc="Input that constrains values to discrete steps.",
    ),
    "switch": _r(
        "switch", _W,
        superclasses=frozenset({"checkbox"}),
        required_states=frozenset({"checked"}),
        name_from="contents",
        desc="Checkbox representing on/off values.",
    ),
    "tab": _r(
        "tab", _W,
        superclasses=frozenset({"sectionhead", "widget"}),
        required_context=frozenset({"tablist"}),
        name_from="contents",
        supported_states=frozenset({"selected"}),
        supported_props=frozenset({"posinset", "setsize"}),
        desc="Grouping label providing selection mechanism for tabpanel.",
    ),
    "tablist": _r(
        "tablist", _W,
        superclasses=frozenset({"composite"}),
        required_owned=frozenset({"tab"}),
        supported_props=frozenset({"multiselectable", "orientation"}),
        supported_states=frozenset({"activedescendant"}),
        desc="List of tab elements.",
    ),
    "tabpanel": _r(
        "tabpanel", _W,
        superclasses=frozenset({"section"}),
        desc="Container for the resource associated with a tab.",
    ),
    "textbox": _r(
        "textbox", _W,
        superclasses=frozenset({"input"}),
        supported_props=frozenset({
            "activedescendant", "autocomplete", "multiline",
            "placeholder", "readonly", "required",
        }),
        desc="Input allowing free-form text.",
    ),
    "tree": _r(
        "tree", _W,
        superclasses=frozenset({"select"}),
        required_owned=frozenset({"group", "treeitem"}),
        supported_props=frozenset({"multiselectable", "orientation"}),
        desc="Widget displaying a hierarchical list.",
    ),
    "treegrid": _r(
        "treegrid", _W,
        superclasses=frozenset({"grid", "tree"}),
        required_owned=frozenset({"row", "rowgroup"}),
        desc="Grid whose rows can be expanded and collapsed.",
    ),
    "treeitem": _r(
        "treeitem", _W,
        superclasses=frozenset({"listitem", "option"}),
        required_context=frozenset({"group", "tree"}),
        name_from="contents",
        supported_states=frozenset({"expanded", "selected"}),
        supported_props=frozenset({"posinset", "setsize"}),
        desc="Item in a tree widget.",
    ),
    "separator": _r(
        "separator", _W,
        superclasses=frozenset({"structure", "widget"}),
        supported_props=frozenset({
            "orientation", "valuemax", "valuemin", "valuenow", "valuetext",
        }),
        desc="Divider between sections (focusable when in a toolbar).",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Document-structure roles (§5.3.3) — 25 roles
# ═══════════════════════════════════════════════════════════════════════════

_DOC_ROLES: Dict[str, AriaRole] = {
    "application": _r(
        "application", _D,
        superclasses=frozenset({"structure"}),
        supported_states=frozenset({"activedescendant"}),
        desc="Region declared as a web application (keyboard handling differs).",
    ),
    "article": _r(
        "article", _D,
        superclasses=frozenset({"document"}),
        supported_props=frozenset({"posinset", "setsize"}),
        desc="Self-contained composition (e.g. blog post, comment).",
    ),
    "blockquote": _r(
        "blockquote", _D,
        superclasses=frozenset({"section"}),
        desc="Section of content quoted from another source.",
    ),
    "caption": _r(
        "caption", _D,
        superclasses=frozenset({"section"}),
        required_context=frozenset({"figure", "grid", "table", "treegrid"}),
        name_from="prohibited",
        desc="Visible label or caption for a figure, table, or grid.",
    ),
    "cell": _r(
        "cell", _D,
        superclasses=frozenset({"section"}),
        required_context=frozenset({"row"}),
        name_from="contents",
        supported_props=frozenset({"colspan", "rowspan", "colindex", "rowindex"}),
        desc="Cell in a tabular container.",
    ),
    "code": _r(
        "code", _D,
        superclasses=frozenset({"section"}),
        desc="Fragment of computer code.",
    ),
    "columnheader": _r(
        "columnheader", _D,
        superclasses=frozenset({"cell", "gridcell", "sectionhead"}),
        required_context=frozenset({"row"}),
        name_from="contents",
        supported_props=frozenset({"sort"}),
        desc="Header cell for a column.",
    ),
    "definition": _r(
        "definition", _D,
        superclasses=frozenset({"section"}),
        desc="Definition of a term or concept.",
    ),
    "deletion": _r(
        "deletion", _D,
        superclasses=frozenset({"section"}),
        desc="Content that has been deleted or marked for removal.",
    ),
    "directory": _r(
        "directory", _D,
        superclasses=frozenset({"list"}),
        desc="List of references to members of a group (deprecated).",
    ),
    "document": _r(
        "document", _D,
        superclasses=frozenset({"structure"}),
        desc="Element containing document content (reading mode).",
    ),
    "emphasis": _r(
        "emphasis", _D,
        superclasses=frozenset({"section"}),
        desc="Content stressed for emphasis.",
    ),
    "feed": _r(
        "feed", _D,
        superclasses=frozenset({"list"}),
        required_owned=frozenset({"article"}),
        desc="Scrollable list of articles.",
    ),
    "figure": _r(
        "figure", _D,
        superclasses=frozenset({"section"}),
        desc="Perceivable section containing graphical document or media.",
    ),
    "generic": _r(
        "generic", _D,
        superclasses=frozenset({"structure"}),
        name_from="prohibited",
        desc="Nameless container with no semantic meaning (div, span).",
    ),
    "group": _r(
        "group", _D,
        superclasses=frozenset({"section"}),
        supported_states=frozenset({"activedescendant"}),
        desc="Set of UI objects not intended for page summary.",
    ),
    "heading": _r(
        "heading", _D,
        superclasses=frozenset({"sectionhead"}),
        required_props=frozenset({"level"}),
        name_from="contents",
        desc="Heading for a section of the page.",
    ),
    "img": _r(
        "img", _D,
        superclasses=frozenset({"section"}),
        desc="Image container (one or more img elements).",
    ),
    "insertion": _r(
        "insertion", _D,
        superclasses=frozenset({"section"}),
        desc="Content that has been inserted.",
    ),
    "list": _r(
        "list", _D,
        superclasses=frozenset({"section"}),
        required_owned=frozenset({"listitem"}),
        desc="Section containing listitem elements.",
    ),
    "listitem": _r(
        "listitem", _D,
        superclasses=frozenset({"section"}),
        required_context=frozenset({"directory", "list"}),
        name_from="contents",
        supported_props=frozenset({"level", "posinset", "setsize"}),
        desc="Single item in a list.",
    ),
    "math": _r(
        "math", _D,
        superclasses=frozenset({"section"}),
        desc="Mathematical expression.",
    ),
    "meter": _r(
        "meter", _D,
        superclasses=frozenset({"range"}),
        supported_props=frozenset({"valuemax", "valuemin", "valuenow", "valuetext"}),
        desc="Scalar measurement within a known range.",
    ),
    "none": _r(
        "none", _D,
        superclasses=frozenset({"structure"}),
        is_presentational=True,
        name_from="prohibited",
        desc="Alias for presentation — no accessible semantics.",
    ),
    "note": _r(
        "note", _D,
        superclasses=frozenset({"section"}),
        desc="Parenthetic or ancillary content.",
    ),
    "paragraph": _r(
        "paragraph", _D,
        superclasses=frozenset({"section"}),
        desc="Paragraph of content.",
    ),
    "presentation": _r(
        "presentation", _D,
        superclasses=frozenset({"structure"}),
        is_presentational=True,
        name_from="prohibited",
        desc="Element with implicit native role removed.",
    ),
    "row": _r(
        "row", _D,
        superclasses=frozenset({"group", "widget"}),
        required_context=frozenset({"grid", "rowgroup", "table", "treegrid"}),
        required_owned=frozenset({"cell", "columnheader", "gridcell", "rowheader"}),
        name_from="contents",
        supported_props=frozenset({
            "colindex", "level", "rowindex", "selected", "expanded",
        }),
        desc="Row of cells in a tabular container.",
    ),
    "rowgroup": _r(
        "rowgroup", _D,
        superclasses=frozenset({"structure"}),
        required_context=frozenset({"grid", "table", "treegrid"}),
        required_owned=frozenset({"row"}),
        desc="Group of rows within a tabular container.",
    ),
    "rowheader": _r(
        "rowheader", _D,
        superclasses=frozenset({"cell", "gridcell", "sectionhead"}),
        required_context=frozenset({"row"}),
        name_from="contents",
        supported_props=frozenset({"sort"}),
        desc="Header cell for a row.",
    ),
    "strong": _r(
        "strong", _D,
        superclasses=frozenset({"section"}),
        desc="Content of strong importance, seriousness, or urgency.",
    ),
    "subscript": _r(
        "subscript", _D,
        superclasses=frozenset({"section"}),
        desc="Subscript content.",
    ),
    "superscript": _r(
        "superscript", _D,
        superclasses=frozenset({"section"}),
        desc="Superscript content.",
    ),
    "table": _r(
        "table", _D,
        superclasses=frozenset({"section"}),
        required_owned=frozenset({"row", "rowgroup"}),
        supported_props=frozenset({"colcount", "rowcount"}),
        desc="Non-interactive tabular data container.",
    ),
    "term": _r(
        "term", _D,
        superclasses=frozenset({"section"}),
        name_from="contents",
        desc="Word or phrase with an optional corresponding definition.",
    ),
    "time": _r(
        "time", _D,
        superclasses=frozenset({"section"}),
        desc="Specific period in time.",
    ),
    "toolbar": _r(
        "toolbar", _D,
        superclasses=frozenset({"group"}),
        supported_props=frozenset({"orientation"}),
        desc="Collection of commonly used function buttons or controls.",
    ),
    "tooltip": _r(
        "tooltip", _D,
        superclasses=frozenset({"section"}),
        name_from="contents",
        desc="Contextual popup displaying a description for an element.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Landmark roles (§5.3.4) — 8 roles
# ═══════════════════════════════════════════════════════════════════════════

_LANDMARK_ROLES: Dict[str, AriaRole] = {
    "banner": _r(
        "banner", _L,
        superclasses=frozenset({"landmark"}),
        desc="Site-oriented content at the beginning of a page.",
    ),
    "complementary": _r(
        "complementary", _L,
        superclasses=frozenset({"landmark"}),
        desc="Supporting content, related to the main content at a similar DOM level.",
    ),
    "contentinfo": _r(
        "contentinfo", _L,
        superclasses=frozenset({"landmark"}),
        desc="Information about the parent document (footer).",
    ),
    "form": _r(
        "form", _L,
        superclasses=frozenset({"landmark"}),
        desc="Landmark region containing a collection of form-associated elements.",
    ),
    "main": _r(
        "main", _L,
        superclasses=frozenset({"landmark"}),
        desc="Primary content of the document.",
    ),
    "navigation": _r(
        "navigation", _L,
        superclasses=frozenset({"landmark"}),
        desc="Collection of navigational elements (links) for the document.",
    ),
    "region": _r(
        "region", _L,
        superclasses=frozenset({"landmark"}),
        desc="Sufficiently important perceivable section that users may want to navigate to.",
    ),
    "search": _r(
        "search", _L,
        superclasses=frozenset({"landmark"}),
        desc="Landmark containing search functionality.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Live-region roles (§5.3.5) — 5 roles
# ═══════════════════════════════════════════════════════════════════════════

_LIVE_ROLES: Dict[str, AriaRole] = {
    "alert": _r(
        "alert", _LR,
        superclasses=frozenset({"section"}),
        supported_props=frozenset({"atomic", "live", "relevant"}),
        desc="Live region with important, usually time-sensitive information.",
    ),
    "log": _r(
        "log", _LR,
        superclasses=frozenset({"section"}),
        supported_props=frozenset({"atomic", "live", "relevant"}),
        desc="Live region where new information is added in meaningful order.",
    ),
    "marquee": _r(
        "marquee", _LR,
        superclasses=frozenset({"section"}),
        supported_props=frozenset({"atomic", "live", "relevant"}),
        desc="Live region with non-essential scrolling information.",
    ),
    "status": _r(
        "status", _LR,
        superclasses=frozenset({"section"}),
        supported_props=frozenset({"atomic", "live", "relevant"}),
        desc="Advisory information that is not important enough for an alert.",
    ),
    "timer": _r(
        "timer", _LR,
        superclasses=frozenset({"status"}),
        supported_props=frozenset({"atomic", "live", "relevant"}),
        desc="Numerical counter indicating elapsed time or remaining time.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Window roles (§5.3.6) — 2 roles
# ═══════════════════════════════════════════════════════════════════════════

_WINDOW_ROLES: Dict[str, AriaRole] = {
    "alertdialog": _r(
        "alertdialog", _WN,
        superclasses=frozenset({"alert", "dialog"}),
        supported_props=frozenset({"modal"}),
        desc="Dialog conveying an alert message requiring immediate user response.",
    ),
    "dialog": _r(
        "dialog", _WN,
        superclasses=frozenset({"window"}),
        supported_props=frozenset({"modal"}),
        desc="Application window designed for user interaction.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════
# Merged taxonomy
# ═══════════════════════════════════════════════════════════════════════════

ROLE_TAXONOMY: Dict[str, AriaRole] = {
    **_ABSTRACT_ROLES,
    **_WIDGET_ROLES,
    **_DOC_ROLES,
    **_LANDMARK_ROLES,
    **_LIVE_ROLES,
    **_WINDOW_ROLES,
}
"""Complete mapping from role name to :class:`AriaRole` for all 82
WAI-ARIA 1.2 roles."""


def get_role(name: str) -> Optional[AriaRole]:
    """Look up a role by name (case-insensitive).

    Parameters:
        name: Role name, e.g. ``"button"`` or ``"NAVIGATION"``.

    Returns:
        The corresponding :class:`AriaRole`, or ``None`` if not found.
    """
    return ROLE_TAXONOMY.get(name.lower())


def role_names() -> Sequence[str]:
    """Return all role names in the taxonomy, sorted alphabetically."""
    return sorted(ROLE_TAXONOMY.keys())


def roles_by_category(category: RoleCategory) -> Sequence[AriaRole]:
    """Return all roles belonging to a given category.

    Parameters:
        category: The desired :class:`RoleCategory`.

    Returns:
        Sequence of :class:`AriaRole` instances in that category,
        sorted by name.
    """
    return sorted(
        (r for r in ROLE_TAXONOMY.values() if r.category == category),
        key=lambda r: r.name,
    )


def is_superclass_of(ancestor: str, descendant: str) -> bool:
    """Check whether *ancestor* is a (transitive) superclass of *descendant*.

    Performs a breadth-first traversal of the ``superclass_roles`` chain.

    Parameters:
        ancestor: Candidate ancestor role name.
        descendant: Candidate descendant role name.

    Returns:
        ``True`` if *ancestor* appears in the superclass chain of
        *descendant*.
    """
    ancestor = ancestor.lower()
    descendant = descendant.lower()
    if ancestor == descendant:
        return True
    role = ROLE_TAXONOMY.get(descendant)
    if role is None:
        return False
    visited: set[str] = set()
    frontier = list(role.superclass_roles)
    while frontier:
        current = frontier.pop()
        if current == ancestor:
            return True
        if current in visited:
            continue
        visited.add(current)
        parent_role = ROLE_TAXONOMY.get(current)
        if parent_role is not None:
            frontier.extend(parent_role.superclass_roles)
    return False
