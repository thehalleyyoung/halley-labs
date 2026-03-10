"""
usability_oracle.benchmarks.generators — Synthetic UI generation.

Creates realistic accessibility trees for benchmarking, including forms,
navigation menus, dashboards, search-result pages, and settings panels.
Each generator produces a well-structured tree with spatial layout.
"""

from __future__ import annotations

import random
import string
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from usability_oracle.accessibility.models import (
    AccessibilityNode,
    AccessibilityState,
    AccessibilityTree,
    BoundingBox,
)
from usability_oracle.core.enums import AccessibilityRole


# ---------------------------------------------------------------------------
# Name pools
# ---------------------------------------------------------------------------

_FIELD_LABELS = [
    "First Name", "Last Name", "Email", "Phone", "Address", "City",
    "State", "Zip Code", "Country", "Date of Birth", "Username",
    "Password", "Confirm Password", "Company", "Job Title", "Website",
    "Bio", "Notes", "Credit Card", "Expiration Date", "CVV",
]

_NAV_LABELS = [
    "Home", "About", "Products", "Services", "Blog", "Contact",
    "Support", "FAQ", "Pricing", "Docs", "Team", "Careers",
    "Settings", "Profile", "Dashboard", "Analytics", "Reports",
    "Notifications", "Messages", "Help",
]

_WIDGET_LABELS = [
    "Revenue", "Users", "Sessions", "Bounce Rate", "Conversion",
    "Active Users", "Page Views", "Orders", "Tickets", "Uptime",
    "Response Time", "Error Rate", "CPU Usage", "Memory", "Disk",
]

_SETTING_LABELS = [
    "Dark Mode", "Notifications", "Email Alerts", "Auto-save",
    "Two-factor Auth", "Language", "Timezone", "Privacy", "Cookies",
    "Analytics", "Backup", "Sync", "Font Size", "Contrast",
    "Keyboard Shortcuts", "Accessibility", "Sound", "Vibration",
]

_BUTTON_LABELS = ["Submit", "Cancel", "Save", "Delete", "Edit", "Next", "Back", "OK", "Close", "Apply"]


def _uid() -> str:
    return uuid.uuid4().hex[:8]


# ---------------------------------------------------------------------------
# SyntheticUIGenerator
# ---------------------------------------------------------------------------

class SyntheticUIGenerator:
    """Generate synthetic accessibility trees for benchmark test cases.

    All randomness is controlled by an optional *seed* for reproducibility.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Forms
    # ------------------------------------------------------------------

    def generate_form(self, n_fields: int = 6, complexity: str = "medium") -> AccessibilityTree:
        """Generate a form with *n_fields* input fields.

        Complexity levels: "simple" (one column), "medium" (two columns
        with validation), "complex" (nested fieldsets, conditional fields).
        """
        root_bbox = BoundingBox(x=0, y=0, width=800, height=max(600, n_fields * 80 + 200))
        root = self._make_node("form-root", AccessibilityRole.FORM.value, "Form", root_bbox, depth=0)
        labels = self._rng.sample(_FIELD_LABELS, min(n_fields, len(_FIELD_LABELS)))
        while len(labels) < n_fields:
            labels.append(f"Field {len(labels) + 1}")

        y_offset = 20.0
        col_width = 380 if complexity != "simple" else 760
        cols = 2 if complexity != "simple" else 1

        for i, label in enumerate(labels):
            col = i % cols
            x = 20 + col * 400 if cols == 2 else 20
            if cols == 2 and i > 0 and i % cols == 0:
                y_offset += 70
            elif cols == 1:
                y_offset += 70

            bbox = BoundingBox(x=x, y=y_offset, width=col_width, height=60)
            group = self._make_node(f"field-group-{i}", AccessibilityRole.REGION.value, label, bbox, depth=1)
            lbl_node = self._make_node(f"label-{i}", AccessibilityRole.GENERIC.value, label,
                                       BoundingBox(x=x, y=y_offset, width=col_width, height=20), depth=2)
            inp_node = self._make_node(f"input-{i}", AccessibilityRole.TEXTFIELD.value, label,
                                       BoundingBox(x=x, y=y_offset + 24, width=col_width, height=32), depth=2)
            group.children = [lbl_node, inp_node]

            if complexity == "complex" and i % 3 == 0:
                help_node = self._make_node(f"help-{i}", AccessibilityRole.GENERIC.value, f"Help for {label}",
                                            BoundingBox(x=x, y=y_offset + 58, width=col_width, height=16), depth=2)
                group.children.append(help_node)

            root.children.append(group)

        y_offset += 90
        submit_bbox = BoundingBox(x=20, y=y_offset, width=120, height=40)
        submit = self._make_node("submit-btn", AccessibilityRole.BUTTON.value, "Submit", submit_bbox, depth=1)
        cancel_bbox = BoundingBox(x=160, y=y_offset, width=120, height=40)
        cancel = self._make_node("cancel-btn", AccessibilityRole.BUTTON.value, "Cancel", cancel_bbox, depth=1)
        root.children.extend([submit, cancel])

        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Navigation
    # ------------------------------------------------------------------

    def generate_navigation(self, n_items: int = 8, depth: int = 2) -> AccessibilityTree:
        """Generate a navigation menu with *n_items* top-level items and *depth* nesting."""
        root_bbox = BoundingBox(x=0, y=0, width=250, height=max(400, n_items * 50 * depth))
        root = self._make_node("nav-root", AccessibilityRole.NAVIGATION.value, "Main Navigation", root_bbox, depth=0)
        labels = self._rng.sample(_NAV_LABELS, min(n_items, len(_NAV_LABELS)))
        while len(labels) < n_items:
            labels.append(f"Page {len(labels) + 1}")

        y = 10.0
        for i, label in enumerate(labels):
            item_bbox = BoundingBox(x=10, y=y, width=230, height=36)
            item = self._make_node(f"nav-item-{i}", AccessibilityRole.LINK.value, label, item_bbox, depth=1)
            y += 40
            if depth > 1:
                n_sub = self._rng.randint(1, 4)
                for j in range(n_sub):
                    sub_bbox = BoundingBox(x=30, y=y, width=210, height=30)
                    sub = self._make_node(f"nav-sub-{i}-{j}", AccessibilityRole.LINK.value,
                                          f"{label} > Sub {j + 1}", sub_bbox, depth=2)
                    item.children.append(sub)
                    y += 34
                    if depth > 2:
                        for k in range(self._rng.randint(0, 2)):
                            deep_bbox = BoundingBox(x=50, y=y, width=190, height=26)
                            deep = self._make_node(f"nav-deep-{i}-{j}-{k}", AccessibilityRole.LINK.value,
                                                   f"{label} > Sub {j + 1} > Item {k + 1}", deep_bbox, depth=3)
                            sub.children.append(deep)
                            y += 30
            root.children.append(item)

        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Dashboard
    # ------------------------------------------------------------------

    def generate_dashboard(self, n_widgets: int = 6) -> AccessibilityTree:
        """Generate a dashboard with *n_widgets* metric widgets."""
        cols = min(n_widgets, 3)
        rows = (n_widgets + cols - 1) // cols
        root_bbox = BoundingBox(x=0, y=0, width=1200, height=rows * 300 + 100)
        root = self._make_node("dash-root", AccessibilityRole.REGION.value, "Dashboard", root_bbox, depth=0)

        header = self._make_node("dash-header", AccessibilityRole.HEADING.value, "Dashboard",
                                 BoundingBox(x=20, y=10, width=1160, height=40), depth=1)
        root.children.append(header)

        labels = self._rng.sample(_WIDGET_LABELS, min(n_widgets, len(_WIDGET_LABELS)))
        while len(labels) < n_widgets:
            labels.append(f"Widget {len(labels) + 1}")

        for i, label in enumerate(labels):
            row, col = divmod(i, cols)
            wx = 20 + col * 390
            wy = 70 + row * 280
            wbox = BoundingBox(x=wx, y=wy, width=370, height=260)
            widget = self._make_node(f"widget-{i}", AccessibilityRole.REGION.value, label, wbox, depth=1)

            title = self._make_node(f"widget-title-{i}", AccessibilityRole.HEADING.value, label,
                                    BoundingBox(x=wx + 10, y=wy + 10, width=350, height=24), depth=2)
            value_str = f"{self._rng.uniform(0, 10000):.0f}"
            value = self._make_node(f"widget-value-{i}", AccessibilityRole.GENERIC.value, value_str,
                                    BoundingBox(x=wx + 10, y=wy + 40, width=350, height=50), depth=2)
            chart = self._make_node(f"widget-chart-{i}", AccessibilityRole.IMAGE.value, f"Chart for {label}",
                                    BoundingBox(x=wx + 10, y=wy + 100, width=350, height=140), depth=2)
            widget.children = [title, value, chart]
            root.children.append(widget)

        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Search results
    # ------------------------------------------------------------------

    def generate_search_results(self, n_results: int = 10) -> AccessibilityTree:
        """Generate a search results page with *n_results* result items."""
        root_bbox = BoundingBox(x=0, y=0, width=900, height=n_results * 130 + 150)
        root = self._make_node("search-root", AccessibilityRole.REGION.value, "Search Results", root_bbox, depth=0)

        search_bar = self._make_node("search-bar", AccessibilityRole.TEXTFIELD.value, "Search",
                                     BoundingBox(x=20, y=10, width=700, height=40), depth=1)
        search_btn = self._make_node("search-btn", AccessibilityRole.BUTTON.value, "Search",
                                     BoundingBox(x=730, y=10, width=80, height=40), depth=1)
        root.children.extend([search_bar, search_btn])

        y = 70.0
        for i in range(n_results):
            item_bbox = BoundingBox(x=20, y=y, width=860, height=110)
            item = self._make_node(f"result-{i}", AccessibilityRole.REGION.value, f"Result {i + 1}", item_bbox, depth=1)

            title_link = self._make_node(f"result-title-{i}", AccessibilityRole.LINK.value,
                                         f"Result Title {i + 1}",
                                         BoundingBox(x=20, y=y, width=860, height=24), depth=2)
            snippet = self._make_node(f"result-snippet-{i}", AccessibilityRole.GENERIC.value,
                                      f"Description snippet for result {i + 1}...",
                                      BoundingBox(x=20, y=y + 28, width=860, height=40), depth=2)
            url_text = self._make_node(f"result-url-{i}", AccessibilityRole.GENERIC.value,
                                       f"https://example.com/page-{i + 1}",
                                       BoundingBox(x=20, y=y + 72, width=860, height=18), depth=2)
            item.children = [title_link, snippet, url_text]
            root.children.append(item)
            y += 120

        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Settings page
    # ------------------------------------------------------------------

    def generate_settings_page(self, n_settings: int = 10) -> AccessibilityTree:
        """Generate a settings page with *n_settings* toggle/select options."""
        root_bbox = BoundingBox(x=0, y=0, width=800, height=n_settings * 70 + 150)
        root = self._make_node("settings-root", AccessibilityRole.FORM.value, "Settings", root_bbox, depth=0)

        heading = self._make_node("settings-heading", AccessibilityRole.HEADING.value, "Settings",
                                  BoundingBox(x=20, y=10, width=760, height=36), depth=1)
        root.children.append(heading)

        labels = self._rng.sample(_SETTING_LABELS, min(n_settings, len(_SETTING_LABELS)))
        while len(labels) < n_settings:
            labels.append(f"Setting {len(labels) + 1}")

        y = 60.0
        for i, label in enumerate(labels):
            row_bbox = BoundingBox(x=20, y=y, width=760, height=54)
            row = self._make_node(f"setting-row-{i}", AccessibilityRole.REGION.value, label, row_bbox, depth=1)

            lbl = self._make_node(f"setting-label-{i}", AccessibilityRole.GENERIC.value, label,
                                  BoundingBox(x=20, y=y + 4, width=400, height=20), depth=2)

            if self._rng.random() < 0.6:
                ctrl = self._make_node(f"setting-toggle-{i}", AccessibilityRole.CHECKBOX.value, label,
                                       BoundingBox(x=600, y=y + 4, width=50, height=28), depth=2)
                ctrl.state = AccessibilityState(checked=self._rng.choice([True, False]))
            else:
                ctrl = self._make_node(f"setting-select-{i}", AccessibilityRole.MENU.value, label,
                                       BoundingBox(x=500, y=y + 4, width=260, height=28), depth=2)
                for j in range(self._rng.randint(2, 5)):
                    opt = self._make_node(f"setting-opt-{i}-{j}", AccessibilityRole.MENUITEM.value,
                                          f"Option {j + 1}",
                                          BoundingBox(x=500, y=y + 34 + j * 28, width=260, height=26), depth=3)
                    ctrl.children.append(opt)

            desc = self._make_node(f"setting-desc-{i}", AccessibilityRole.GENERIC.value,
                                   f"Configure {label.lower()} preferences",
                                   BoundingBox(x=20, y=y + 28, width=400, height=16), depth=2)
            row.children = [lbl, ctrl, desc]
            root.children.append(row)
            y += 64

        y += 20
        save_btn = self._make_node("save-btn", AccessibilityRole.BUTTON.value, "Save Changes",
                                   BoundingBox(x=20, y=y, width=140, height=40), depth=1)
        root.children.append(save_btn)

        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Node helpers
    # ------------------------------------------------------------------

    def _random_node(self, role: str, depth: int = 0) -> AccessibilityNode:
        name = self._random_name(role)
        bbox = self._random_bounding_box(BoundingBox(x=0, y=0, width=800, height=600))
        return self._make_node(_uid(), role, name, bbox, depth)

    def _random_bounding_box(self, parent_bbox: BoundingBox) -> BoundingBox:
        max_w = max(parent_bbox.width * 0.8, 20)
        max_h = max(parent_bbox.height * 0.8, 20)
        w = self._rng.uniform(20, max_w)
        h = self._rng.uniform(16, max_h)
        x = parent_bbox.x + self._rng.uniform(0, max(parent_bbox.width - w, 0))
        y = parent_bbox.y + self._rng.uniform(0, max(parent_bbox.height - h, 0))
        return BoundingBox(x=round(x, 1), y=round(y, 1), width=round(w, 1), height=round(h, 1))

    def _random_name(self, role: str) -> str:
        pools: dict[str, list[str]] = {
            AccessibilityRole.BUTTON.value: _BUTTON_LABELS,
            AccessibilityRole.LINK.value: _NAV_LABELS,
            AccessibilityRole.TEXTFIELD.value: _FIELD_LABELS,
            AccessibilityRole.HEADING.value: ["Dashboard", "Settings", "Profile", "Analytics"],
            AccessibilityRole.CHECKBOX.value: _SETTING_LABELS,
        }
        pool = pools.get(role, _NAV_LABELS)
        return self._rng.choice(pool)

    @staticmethod
    def _apply_layout(nodes: list[AccessibilityNode], layout: str, parent_bbox: BoundingBox) -> list[AccessibilityNode]:
        """Reposition *nodes* according to *layout* within *parent_bbox*."""
        if not nodes:
            return nodes
        n = len(nodes)
        if layout == "vertical":
            h = parent_bbox.height / n
            for i, node in enumerate(nodes):
                node.bounding_box = BoundingBox(
                    x=parent_bbox.x, y=parent_bbox.y + i * h,
                    width=parent_bbox.width, height=h,
                )
        elif layout == "horizontal":
            w = parent_bbox.width / n
            for i, node in enumerate(nodes):
                node.bounding_box = BoundingBox(
                    x=parent_bbox.x + i * w, y=parent_bbox.y,
                    width=w, height=parent_bbox.height,
                )
        elif layout == "grid":
            cols = max(1, int(n ** 0.5))
            rows = (n + cols - 1) // cols
            cw = parent_bbox.width / cols
            ch = parent_bbox.height / rows
            for i, node in enumerate(nodes):
                r, c = divmod(i, cols)
                node.bounding_box = BoundingBox(
                    x=parent_bbox.x + c * cw, y=parent_bbox.y + r * ch,
                    width=cw, height=ch,
                )
        # "free" layout leaves nodes as-is
        return nodes

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _make_node(nid: str, role: str, name: str, bbox: BoundingBox, depth: int) -> AccessibilityNode:
        return AccessibilityNode(
            id=nid,
            role=role,
            name=name,
            bounding_box=bbox,
            properties={},
            state=AccessibilityState(),
            children=[],
            depth=depth,
        )

    @staticmethod
    def _build_tree(root: AccessibilityNode) -> AccessibilityTree:
        idx: dict[str, AccessibilityNode] = {}

        def _index(node: AccessibilityNode) -> None:
            idx[node.id] = node
            for child in node.children:
                child.parent_id = node.id
                _index(child)

        _index(root)
        return AccessibilityTree(root=root, node_index=idx)

    # ------------------------------------------------------------------
    # Data-table / data-grid generator
    # ------------------------------------------------------------------

    def generate_data_table(
        self,
        rows: int = 10,
        cols: int = 5,
        sortable: bool = True,
        filterable: bool = False,
    ) -> AccessibilityTree:
        """Generate a data table with column headers and cells.

        Produces a table role with thead/tbody structure, optional
        sortable column headers and filter inputs.
        """
        bbox = BoundingBox(x=0, y=0, width=800, height=600)
        table_root = self._make_node("table-root", "table", "Data Table", bbox, 0)
        children: list[AccessibilityNode] = []

        # Column headers
        header_row = self._make_node("thead-row", "row", "", BoundingBox(x=0, y=0, width=800, height=40), 1)
        for c in range(cols):
            hdr = self._make_node(
                f"col-header-{c}", "columnheader", f"Column {c}",
                BoundingBox(x=c * (800 // cols), y=0, width=800 // cols, height=40), 2,
            )
            if sortable:
                hdr.properties["aria-sort"] = "none"
                btn = self._make_node(
                    f"sort-btn-{c}", "button", f"Sort column {c}",
                    BoundingBox(x=c * (800 // cols) + 60, y=5, width=30, height=30), 3,
                )
                hdr.children.append(btn)
            header_row.children.append(hdr)
        children.append(header_row)

        # Filter row
        if filterable:
            filter_row = self._make_node("filter-row", "row", "", BoundingBox(x=0, y=40, width=800, height=35), 1)
            for c in range(cols):
                flt = self._make_node(
                    f"filter-{c}", "textbox", f"Filter column {c}",
                    BoundingBox(x=c * (800 // cols), y=40, width=800 // cols, height=35), 2,
                )
                filter_row.children.append(flt)
            children.append(filter_row)

        # Data rows
        y_offset = 80 if filterable else 40
        row_height = max(1, (600 - y_offset) // rows)
        for r in range(rows):
            row_node = self._make_node(
                f"row-{r}", "row", "",
                BoundingBox(x=0, y=y_offset + r * row_height, width=800, height=row_height), 1,
            )
            for c in range(cols):
                cell = self._make_node(
                    f"cell-{r}-{c}", "cell", f"R{r}C{c}",
                    BoundingBox(x=c * (800 // cols), y=y_offset + r * row_height, width=800 // cols, height=row_height), 2,
                )
                row_node.children.append(cell)
            children.append(row_node)

        table_root.children = children
        return self._build_tree(table_root)

    # ------------------------------------------------------------------
    # Wizard / multi-step flow generator
    # ------------------------------------------------------------------

    def generate_wizard(
        self,
        n_steps: int = 4,
        fields_per_step: int = 3,
    ) -> AccessibilityTree:
        """Generate a multi-step wizard form.

        Each step has a heading, progress indicator, form fields,
        and navigation buttons (Back / Next / Submit).
        """
        bbox = BoundingBox(x=0, y=0, width=600, height=800)
        root = self._make_node("wizard-root", "main", "Wizard", bbox, 0)
        children: list[AccessibilityNode] = []

        # Progress indicator
        progress = self._make_node(
            "progress", "progressbar", f"Step 1 of {n_steps}",
            BoundingBox(x=0, y=0, width=600, height=30), 1,
        )
        progress.properties["aria-valuemin"] = "0"
        progress.properties["aria-valuemax"] = str(n_steps)
        progress.properties["aria-valuenow"] = "1"
        children.append(progress)

        # Steps
        step_height = (800 - 100) // n_steps
        for s in range(n_steps):
            step = self._make_node(
                f"step-{s}", "group", f"Step {s + 1}",
                BoundingBox(x=0, y=40 + s * step_height, width=600, height=step_height), 1,
            )
            # Heading
            heading = self._make_node(
                f"step-heading-{s}", "heading", f"Step {s + 1}: Details",
                BoundingBox(x=10, y=40 + s * step_height, width=580, height=30), 2,
            )
            step.children.append(heading)

            # Fields
            field_h = max(1, (step_height - 80) // fields_per_step)
            for f_idx in range(fields_per_step):
                label = self._make_node(
                    f"label-{s}-{f_idx}", "label", f"Field {f_idx + 1}",
                    BoundingBox(x=10, y=80 + s * step_height + f_idx * field_h, width=150, height=field_h), 2,
                )
                inp = self._make_node(
                    f"input-{s}-{f_idx}", "textbox", f"Field {f_idx + 1}",
                    BoundingBox(x=170, y=80 + s * step_height + f_idx * field_h, width=400, height=field_h), 2,
                )
                inp.properties["aria-required"] = "true" if f_idx == 0 else "false"
                step.children.extend([label, inp])

            # Only first step visible
            if s > 0:
                step.state.hidden = True
            children.append(step)

        # Navigation buttons
        nav = self._make_node("nav-buttons", "group", "", BoundingBox(x=0, y=760, width=600, height=40), 1)
        back_btn = self._make_node("btn-back", "button", "Back", BoundingBox(x=10, y=760, width=80, height=36), 2)
        back_btn.state.disabled = True  # disabled on first step
        next_btn = self._make_node("btn-next", "button", "Next", BoundingBox(x=100, y=760, width=80, height=36), 2)
        submit_btn = self._make_node("btn-submit", "button", "Submit", BoundingBox(x=500, y=760, width=90, height=36), 2)
        submit_btn.state.hidden = True  # hidden until last step
        nav.children = [back_btn, next_btn, submit_btn]
        children.append(nav)

        root.children = children
        return self._build_tree(root)

    # ------------------------------------------------------------------
    # Modal dialog generator
    # ------------------------------------------------------------------

    def generate_modal_dialog(
        self,
        n_actions: int = 3,
        has_close_button: bool = True,
    ) -> AccessibilityTree:
        """Generate a modal dialog with title, content, and action buttons."""
        bbox = BoundingBox(x=100, y=100, width=400, height=300)
        root = self._make_node("dialog-root", "dialog", "Confirmation", bbox, 0)
        root.properties["aria-modal"] = "true"

        children: list[AccessibilityNode] = []

        # Title
        title = self._make_node(
            "dialog-title", "heading", "Confirm Action",
            BoundingBox(x=110, y=110, width=380, height=30), 1,
        )
        children.append(title)
        root.properties["aria-labelledby"] = "dialog-title"

        # Close button
        if has_close_button:
            close_btn = self._make_node(
                "close-btn", "button", "Close",
                BoundingBox(x=470, y=105, width=24, height=24), 1,
            )
            close_btn.properties["aria-label"] = "Close dialog"
            children.append(close_btn)

        # Content
        content = self._make_node(
            "dialog-content", "text", "Are you sure you want to proceed?",
            BoundingBox(x=110, y=150, width=380, height=100), 1,
        )
        children.append(content)

        # Action buttons
        btn_width = max(1, 380 // n_actions)
        for i in range(n_actions):
            names = ["Cancel", "OK", "Apply", "Save", "Delete"]
            label = names[i] if i < len(names) else f"Action {i}"
            btn = self._make_node(
                f"action-btn-{i}", "button", label,
                BoundingBox(x=110 + i * btn_width, y=360, width=btn_width - 10, height=30), 1,
            )
            children.append(btn)

        root.children = children
        return self._build_tree(root)
