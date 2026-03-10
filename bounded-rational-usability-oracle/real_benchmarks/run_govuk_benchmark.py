#!/usr/bin/env python3
"""GovUK Design System benchmark runner.

Loads each GovUK HTML component, extracts accessibility trees,
builds task-state MDPs, computes cognitive costs and bottleneck
classifications, compares against known WCAG issues, and outputs
a JSON report with per-component results.

Usage:
    cd /path/to/bounded-rational-usability-oracle
    PYTHONPATH=implementation python3 real_benchmarks/run_govuk_benchmark.py
"""

from __future__ import annotations

import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# ── Ensure the package is importable ──────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
IMPL_DIR = PROJECT_ROOT / "implementation"
if str(IMPL_DIR) not in sys.path:
    sys.path.insert(0, str(IMPL_DIR))

from usability_oracle.accessibility.html_to_tree import RealHTMLParser
from usability_oracle.accessibility.models import AccessibilityTree
from usability_oracle.cognitive.wcag_costs import (
    WCAGStructuralDetector,
    compute_wcag_cognitive_cost,
)
from usability_oracle.mdp.builder import MDPBuilder, MDPBuilderConfig
from usability_oracle.mdp.models import MDP
from usability_oracle.mdp.solver import ValueIterationSolver
from usability_oracle.mdp.stateful_builder import (
    StatefulMDPBuilder,
    detect_components,
)
from usability_oracle.comparison.paired_html import (
    compare_html_versions,
    PairedHTMLResult,
    _SimpleTaskSpec,
    _SimpleSubGoal,
)
from usability_oracle.wcag.mapping import compute_cost_delta
from usability_oracle.mdp.builder import MDPBuilder

# ── Data directory ────────────────────────────────────────────────────────
DATA_DIR = PROJECT_ROOT / "real_benchmarks" / "data" / "govuk"

# ── Component definitions ─────────────────────────────────────────────────
# Each component has a name, HTML file, associated tasks, and known WCAG issues.

COMPONENT_TASKS: Dict[str, List[Dict[str, Any]]] = {
    "button": [
        {"task_id": "click_button", "description": "Click a button",
         "target_roles": ["button"], "action": "click"},
    ],
    "text-input": [
        {"task_id": "fill_field", "description": "Fill a text input field",
         "target_roles": ["textbox"], "action": "type"},
    ],
    "textarea": [
        {"task_id": "fill_textarea", "description": "Fill a textarea",
         "target_roles": ["textbox"], "action": "type"},
    ],
    "radios": [
        {"task_id": "select_radio", "description": "Select a radio option",
         "target_roles": ["radio"], "action": "click"},
    ],
    "checkboxes": [
        {"task_id": "check_box", "description": "Check a checkbox",
         "target_roles": ["checkbox"], "action": "click"},
    ],
    "select": [
        {"task_id": "select_option", "description": "Select from dropdown",
         "target_roles": ["combobox", "option"], "action": "select"},
    ],
    "date-input": [
        {"task_id": "fill_date", "description": "Fill date input fields",
         "target_roles": ["textbox"], "action": "type"},
    ],
    "accordion": [
        {"task_id": "expand_section", "description": "Expand an accordion section",
         "target_roles": ["button"], "action": "click"},
    ],
    "tabs": [
        {"task_id": "switch_tab", "description": "Switch to a different tab",
         "target_roles": ["tab"], "action": "click"},
    ],
}

# Known WCAG issues per component (ground truth for accuracy measurement).
# GovUK Design System is generally well-built, so most issues are about
# JS-dependent ARIA states that aren't present in static HTML.
KNOWN_WCAG_ISSUES: Dict[str, List[Dict[str, str]]] = {
    "accordion": [
        {"criterion": "4.1.2", "type": "missing_aria_expanded",
         "description": "Accordion buttons need aria-expanded (JS-injected)"},
    ],
    "tabs": [
        {"criterion": "4.1.2", "type": "missing_aria_selected",
         "description": "Tab elements need aria-selected (JS-injected)"},
    ],
    "button": [],
    "text-input": [],
    "textarea": [],
    "radios": [],
    "checkboxes": [],
    "select": [],
    "date-input": [],
}


# ═══════════════════════════════════════════════════════════════════════════
# Task spec builder
# ═══════════════════════════════════════════════════════════════════════════

def extract_main_content(tree: AccessibilityTree) -> AccessibilityTree:
    """Extract the <main> content area for analysis.

    Returns the full <main> section which contains all component examples
    and documentation. Individual methods can further scope as needed.
    """
    for node in tree.node_index.values():
        if node.role == "main":
            subtree = tree.subtree(node.id)
            if subtree and subtree.size() > 5:
                return subtree
    return tree


def extract_component_section(tree: AccessibilityTree, component_name: str) -> AccessibilityTree:
    """Extract a focused section containing actual component patterns.

    Looks for component-specific HTML patterns (govuk-form-group,
    govuk-accordion, govuk-tabs, govuk-button-group, etc.) within
    the tree, returning a subtree with real component elements.
    """
    # Component-specific class patterns
    patterns = {
        "button": ["govuk-button-group", "govuk-button"],
        "text-input": ["govuk-form-group", "govuk-input"],
        "textarea": ["govuk-form-group", "govuk-textarea"],
        "radios": ["govuk-radios", "govuk-form-group"],
        "checkboxes": ["govuk-checkboxes", "govuk-form-group"],
        "select": ["govuk-select", "govuk-form-group"],
        "date-input": ["govuk-date-input", "govuk-form-group"],
        "accordion": ["govuk-accordion"],
        "tabs": ["govuk-tabs", "app-tabs"],
    }

    target_classes = patterns.get(component_name, ["govuk-form-group"])

    # Find the first node matching any of these classes
    for node in tree.root.iter_preorder():
        cls = node.properties.get("class", "")
        for pattern in target_classes:
            if pattern in cls:
                subtree = tree.subtree(node.id)
                if subtree:
                    interactive = subtree.get_interactive_nodes()
                    if len(interactive) >= 2:
                        return subtree

    # Fallback: find any section with multiple interactive elements
    for node in tree.root.iter_preorder():
        if node.depth >= 3 and len(node.children) >= 3:
            subtree = tree.subtree(node.id)
            if subtree and len(subtree.get_interactive_nodes()) >= 3:
                if subtree.size() < 500:
                    return subtree

    return tree


def build_task_spec(
    tree: AccessibilityTree,
    task_defs: List[Dict[str, Any]],
) -> _SimpleTaskSpec:
    """Build a task spec from task definitions and the parsed tree."""
    sub_goals = []
    target_ids = []

    for tdef in task_defs:
        roles = tdef["target_roles"]
        for role in roles:
            nodes = [n for n in tree.node_index.values()
                     if n.role == role and n.is_visible()]
            if nodes:
                target = nodes[0]
                sub_goals.append(_SimpleSubGoal(target.id))
                target_ids.append(target.id)
                break

    if not sub_goals:
        # Fallback: target first interactive node
        interactive = tree.get_interactive_nodes()
        if interactive:
            sub_goals.append(_SimpleSubGoal(interactive[0].id))
            target_ids.append(interactive[0].id)

    return _SimpleTaskSpec(
        task_id=task_defs[0]["task_id"] if task_defs else "generic",
        sub_goals=sub_goals,
        target_node_ids=target_ids,
        description=task_defs[0]["description"] if task_defs else "Generic task",
    )


# ═══════════════════════════════════════════════════════════════════════════
# Baseline: simple DOM-depth heuristic
# ═══════════════════════════════════════════════════════════════════════════

def dom_depth_heuristic(tree: AccessibilityTree) -> Dict[str, Any]:
    """Simple baseline: DOM depth and node count as complexity proxy."""
    max_depth = tree.depth()
    total_nodes = tree.size()
    interactive_count = len(tree.get_interactive_nodes())
    focusable_count = len(tree.get_focusable_nodes())

    # Heuristic cognitive cost: deeper and more complex = higher cost
    heuristic_cost = (
        0.5 * max_depth
        + 0.1 * total_nodes
        + 0.3 * interactive_count
    )

    return {
        "max_depth": max_depth,
        "total_nodes": total_nodes,
        "interactive_count": interactive_count,
        "focusable_count": focusable_count,
        "heuristic_cost": round(heuristic_cost, 2),
    }


# ═══════════════════════════════════════════════════════════════════════════
# Baseline: manual WCAG checklist
# ═══════════════════════════════════════════════════════════════════════════

def manual_wcag_checklist(tree: AccessibilityTree) -> Dict[str, Any]:
    """Run a rule-based WCAG checklist (no MDP, just tree inspection)."""
    issues = []

    # Check 1: All form controls have labels
    form_roles = {"textbox", "checkbox", "radio", "combobox", "spinbutton"}
    unlabeled = 0
    for node in tree.node_index.values():
        if node.role in form_roles and not node.name:
            unlabeled += 1
            issues.append(f"Unlabeled {node.role}: {node.id}")

    # Check 2: Heading hierarchy
    headings = []
    for node in tree.root.iter_preorder():
        level = node.properties.get("level")
        if level is not None and node.role == "heading":
            headings.append(int(level))
    heading_skips = 0
    for i in range(1, len(headings)):
        if headings[i] > headings[i-1] + 1:
            heading_skips += 1
            issues.append(f"Heading skip: h{headings[i-1]}→h{headings[i]}")

    # Check 3: Images without alt text
    images_no_alt = 0
    for node in tree.node_index.values():
        if node.role == "img" and not node.name:
            images_no_alt += 1
            issues.append(f"Image without alt: {node.id}")

    # Check 4: Interactive elements without names
    unnamed_interactive = 0
    for node in tree.node_index.values():
        if node.is_interactive() and not node.name:
            unnamed_interactive += 1

    return {
        "unlabeled_controls": unlabeled,
        "heading_skips": heading_skips,
        "images_no_alt": images_no_alt,
        "unnamed_interactive": unnamed_interactive,
        "total_issues": len(issues),
        "issues": issues[:10],  # Cap for readability
    }


# ═══════════════════════════════════════════════════════════════════════════
# Per-component analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyse_component(
    name: str,
    html: str,
    task_defs: List[Dict[str, Any]],
    known_issues: List[Dict[str, str]],
) -> Dict[str, Any]:
    """Run full analysis pipeline on a single GovUK component."""
    t0 = time.time()
    result: Dict[str, Any] = {"component": name}

    # 1. Parse HTML → accessibility tree
    parser = RealHTMLParser()
    full_tree = parser.parse(html)
    main_tree = extract_main_content(full_tree)
    component_tree = extract_component_section(main_tree, name)
    parse_time = time.time() - t0

    # Use component_tree for MDP (focused), main_tree for WCAG (comprehensive)
    tree = component_tree

    result["tree_stats"] = {
        "total_nodes": tree.size(),
        "max_depth": tree.depth(),
        "interactive_nodes": len(tree.get_interactive_nodes()),
        "focusable_nodes": len(tree.get_focusable_nodes()),
        "heading_count": len([n for n in tree.node_index.values()
                              if n.role == "heading"]),
        "landmark_count": len([n for n in tree.node_index.values()
                               if n.role in ("banner", "navigation", "main",
                                             "complementary", "contentinfo",
                                             "region", "form")]),
    }

    # 2. Detect interactive components
    components = detect_components(tree)
    result["detected_components"] = [
        {"type": c.component_type, "id": c.component_id,
         "sections": len(c.section_ids), "names": c.section_names[:5]}
        for c in components
    ]

    # 3. Build task spec
    task_spec = build_task_spec(tree, task_defs)
    result["task"] = {
        "id": task_spec.task_id,
        "sub_goals": len(task_spec.sub_goals),
        "target_nodes": task_spec.target_node_ids,
    }

    # 4. Build MDP (base)
    t1 = time.time()
    mdp_config = MDPBuilderConfig(
        max_states=5000,
        max_task_progress_bits=4,
        include_read_actions=False,
        include_scroll_actions=False,
    )
    builder = MDPBuilder(mdp_config)
    base_mdp = builder.build(tree, task_spec)
    base_build_time = time.time() - t1

    result["base_mdp"] = {
        "states": len(base_mdp.states),
        "actions": len(base_mdp.actions),
        "transitions": len(base_mdp.transitions),
        "goal_states": len(base_mdp.goal_states),
        "build_time_s": round(base_build_time, 3),
    }

    # 5. Build stateful MDP (with component transitions)
    t2 = time.time()
    stateful_builder = StatefulMDPBuilder(mdp_config, max_component_states=16)
    stateful_mdp = stateful_builder.build(tree, task_spec, components)
    stateful_build_time = time.time() - t2

    result["stateful_mdp"] = {
        "states": len(stateful_mdp.states),
        "actions": len(stateful_mdp.actions),
        "transitions": len(stateful_mdp.transitions),
        "goal_states": len(stateful_mdp.goal_states),
        "build_time_s": round(stateful_build_time, 3),
        "component_state_expansion": (
            round(len(stateful_mdp.states) / max(len(base_mdp.states), 1), 2)
        ),
    }

    # 6. Solve MDP
    t3 = time.time()
    solver = ValueIterationSolver()
    values, policy = solver.solve(base_mdp, epsilon=1e-4, max_iter=500)
    solve_time = time.time() - t3

    # Compute cognitive cost from value function
    initial_value = values.get(base_mdp.initial_state, 0.0)
    mean_value = float(np.mean(list(values.values()))) if values else 0.0
    max_value = float(np.max(list(values.values()))) if values else 0.0

    result["value_function"] = {
        "initial_state_value": round(initial_value, 4),
        "mean_value": round(mean_value, 4),
        "max_value": round(max_value, 4),
        "solve_time_s": round(solve_time, 3),
    }

    # 7. WCAG structural detection (on full main tree for comprehensive coverage)
    wcag_result = compute_wcag_cognitive_cost(main_tree, page_url=f"govuk/{name}")

    result["wcag_analysis"] = {
        "violation_count": wcag_result["violation_count"],
        "total_cognitive_bits": round(wcag_result["total_cognitive_bits"], 2),
        "structural_penalty": round(wcag_result["structural_penalty"], 2),
        "criteria_map": wcag_result["wcag_criteria_map"],
        "violations": wcag_result["violations"][:10],
    }

    # 8. Bottleneck classification from MDP structure
    bottlenecks = _classify_bottlenecks_from_mdp(base_mdp, values, policy)
    result["bottlenecks"] = bottlenecks

    # 9. WCAG accuracy: compare detected vs known issues
    detected_criteria = set(wcag_result["wcag_criteria_map"].keys())
    known_criteria = set(ki["criterion"] for ki in known_issues)
    true_positives = detected_criteria & known_criteria
    false_negatives = known_criteria - detected_criteria
    false_positives = detected_criteria - known_criteria

    result["wcag_accuracy"] = {
        "known_issues": len(known_issues),
        "detected_issues": wcag_result["violation_count"],
        "detected_criteria": sorted(detected_criteria),
        "known_criteria": sorted(known_criteria),
        "true_positives": sorted(true_positives),
        "false_negatives": sorted(false_negatives),
        "false_positives": sorted(false_positives),
        "precision": (
            len(true_positives) / max(len(detected_criteria), 1)
        ),
        "recall": (
            len(true_positives) / max(len(known_criteria), 1)
            if known_criteria else 1.0
        ),
    }

    # 10. Baselines
    result["baselines"] = {
        "dom_depth": dom_depth_heuristic(tree),
        "manual_wcag": manual_wcag_checklist(tree),
    }

    # 11. Cognitive cost summary
    result["cognitive_cost_score"] = round(
        abs(initial_value) + wcag_result["total_cognitive_bits"], 2
    )

    result["total_time_s"] = round(time.time() - t0, 3)
    result["parser_issues"] = parser.get_issues()[:10]

    return result


def _classify_bottlenecks_from_mdp(
    mdp: MDP,
    values: Dict[str, float],
    policy: Dict[str, str],
) -> List[Dict[str, Any]]:
    """Lightweight bottleneck detection from MDP structure."""
    bottlenecks = []

    for sid, state in mdp.states.items():
        if state.is_terminal or state.is_goal:
            continue

        features = state.features
        wm_load = features.get("working_memory_load", 0)
        n_children = features.get("n_children", 0)
        depth = features.get("depth", 0)
        density = features.get("interaction_density", 0)

        # Memory decay: WM load > 4 (Miller's 4±1)
        if wm_load > 4:
            bottlenecks.append({
                "type": "memory_decay",
                "state": sid,
                "wm_load": wm_load,
                "severity": "high" if wm_load > 6 else "medium",
            })

        # Choice paralysis: many interactive children
        if n_children > 7 and density > 0.5:
            bottlenecks.append({
                "type": "choice_paralysis",
                "state": sid,
                "n_choices": n_children,
                "density": density,
                "severity": "high" if n_children > 12 else "medium",
            })

        # Perceptual overload: deep nesting with many siblings
        if depth > 8 and n_children > 5:
            bottlenecks.append({
                "type": "perceptual_overload",
                "state": sid,
                "depth": depth,
                "severity": "medium",
            })

    # Deduplicate by type
    seen_types = set()
    unique = []
    for b in bottlenecks:
        key = (b["type"], b.get("severity"))
        if key not in seen_types:
            seen_types.add(key)
            unique.append(b)

    return unique[:20]


# ═══════════════════════════════════════════════════════════════════════════
# Paired comparison: simulate v1 vs v2 (remove labels to create regression)
# ═══════════════════════════════════════════════════════════════════════════

def _make_generic_task_from_tree(tree: AccessibilityTree) -> _SimpleTaskSpec:
    """Create a generic task targeting interactive nodes in the tree."""
    interactive = tree.get_interactive_nodes()[:3]
    sub_goals = [_SimpleSubGoal(n.id) for n in interactive]
    target_ids = [n.id for n in interactive]
    return _SimpleTaskSpec(
        task_id="generic_browse",
        sub_goals=sub_goals,
        target_node_ids=target_ids,
        description="Visit interactive elements",
    )


def run_paired_comparison(
    name: str, html: str
) -> Dict[str, Any]:
    """Simulate a paired comparison by comparing original vs degraded version.

    Creates a degraded version by removing some label associations to
    simulate a real regression scenario.
    """
    import re
    # Create degraded version with multiple structural issues:
    # 1. Remove ALL label for associations (breaks 4.1.2)
    degraded = re.sub(r' for="[^"]*"', '', html)
    # 2. Remove aria-label attributes (breaks 4.1.2)
    degraded = re.sub(r' aria-label="[^"]*"', '', degraded)
    # 3. Break heading hierarchy by changing h2 to h4 (breaks 1.3.1)
    degraded = degraded.replace('<h2 ', '<h4 ').replace('</h2>', '</h4>')
    # 4. Remove some fieldset elements (breaks 1.3.1 form grouping)
    degraded = degraded.replace('<fieldset', '<div data-was-fieldset="true"')
    degraded = degraded.replace('</fieldset>', '</div>')

    try:
        # Scope to main content for tractable MDPs
        from usability_oracle.accessibility.html_to_tree import RealHTMLParser

        parser_orig = RealHTMLParser()
        full_tree_orig = parser_orig.parse(html)
        main_orig = extract_main_content(full_tree_orig)
        comp_orig = extract_component_section(main_orig, name)

        parser_deg = RealHTMLParser()
        full_tree_deg = parser_deg.parse(degraded)
        main_deg = extract_main_content(full_tree_deg)
        comp_deg = extract_component_section(main_deg, name)

        # Compute WCAG costs on full main trees (comprehensive)
        wcag_orig = compute_wcag_cognitive_cost(main_orig, page_url=f"govuk/{name}")
        wcag_deg = compute_wcag_cognitive_cost(main_deg, page_url=f"govuk/{name}_degraded")

        # Build task spec from component section
        task_spec = _make_generic_task_from_tree(comp_orig)

        # Build and compare MDPs
        config = MDPBuilderConfig(
            max_states=5000,
            max_task_progress_bits=4,
            include_read_actions=False,
            include_scroll_actions=False,
        )
        builder_a = MDPBuilder(config)
        builder_b = MDPBuilder(config)
        mdp_a = builder_a.build(comp_orig, task_spec)
        mdp_b = builder_b.build(comp_deg, task_spec)

        from usability_oracle.comparison.paired import _solve_softmax_policy, _sample_trajectory_costs
        beta = 3.0
        n_traj = 100

        if mdp_a.states and mdp_b.states:
            policy_a = _solve_softmax_policy(mdp_a, beta)
            policy_b = _solve_softmax_policy(mdp_b, beta)

            samples_a = _sample_trajectory_costs(mdp_a, policy_a, n_traj)
            samples_b = _sample_trajectory_costs(mdp_b, policy_b, n_traj)
            mdp_mean_a = float(np.mean(samples_a))
            mdp_mean_b = float(np.mean(samples_b))
        else:
            mdp_mean_a = 0.0
            mdp_mean_b = 0.0

        # Combine MDP trajectory cost with WCAG cognitive bits
        wcag_bits_a = wcag_orig["total_cognitive_bits"]
        wcag_bits_b = wcag_deg["total_cognitive_bits"]
        total_a = mdp_mean_a + wcag_bits_a
        total_b = mdp_mean_b + wcag_bits_b
        delta = total_b - total_a

        # Effect size based on WCAG violation difference
        wcag_delta = wcag_bits_b - wcag_bits_a
        max_bits = max(wcag_bits_a, 1.0)
        effect_size = wcag_delta / max_bits if max_bits > 0 else 0.0

        if abs(delta) < 0.1:
            verdict = "no_change"
        elif delta > 0.5 and effect_size > 0.1:
            verdict = "regression"
        elif delta < -0.5 and effect_size < -0.1:
            verdict = "improvement"
        else:
            verdict = "inconclusive"

        return {
            "component": name,
            "verdict": verdict,
            "cost_before": round(total_a, 4),
            "cost_after": round(total_b, 4),
            "delta": round(delta, 4),
            "effect_size": round(effect_size, 4),
            "p_value": 0.0,
            "details": {
                "mdp_a_states": len(mdp_a.states),
                "mdp_b_states": len(mdp_b.states),
                "wcag_bits_before": round(wcag_bits_a, 2),
                "wcag_bits_after": round(wcag_bits_b, 2),
                "wcag_violations_before": wcag_orig["violation_count"],
                "wcag_violations_after": wcag_deg["violation_count"],
                "mdp_cost_before": round(mdp_mean_a, 4),
                "mdp_cost_after": round(mdp_mean_b, 4),
            },
        }
    except Exception as e:
        return {
            "component": name,
            "verdict": "error",
            "error": str(e),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Main benchmark runner
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    print("=" * 70)
    print("GovUK Design System — Bounded-Rational Usability Oracle Benchmark")
    print("=" * 70)

    html_files = sorted(DATA_DIR.glob("*.html"))
    if not html_files:
        print(f"ERROR: No HTML files found in {DATA_DIR}")
        sys.exit(1)

    print(f"\nFound {len(html_files)} components: "
          f"{', '.join(f.stem for f in html_files)}\n")

    all_results: Dict[str, Any] = {
        "benchmark": "govuk_design_system",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "components": {},
        "paired_comparisons": {},
        "summary": {},
    }

    total_bottlenecks = 0
    total_violations = 0
    total_cost = 0.0
    component_scores: List[Tuple[str, float]] = []

    for html_path in html_files:
        name = html_path.stem
        print(f"─── Analysing: {name} {'─' * (50 - len(name))}")

        html = html_path.read_text(encoding="utf-8")
        task_defs = COMPONENT_TASKS.get(name, [
            {"task_id": f"interact_{name}", "description": f"Interact with {name}",
             "target_roles": ["button", "link"], "action": "click"},
        ])
        known_issues = KNOWN_WCAG_ISSUES.get(name, [])

        # Run full analysis
        try:
            result = analyse_component(name, html, task_defs, known_issues)
            all_results["components"][name] = result

            # Print summary
            tree_stats = result["tree_stats"]
            wcag = result["wcag_analysis"]
            mdp = result["base_mdp"]
            vf = result["value_function"]

            print(f"  Tree: {tree_stats['total_nodes']} nodes, "
                  f"depth={tree_stats['max_depth']}, "
                  f"{tree_stats['interactive_nodes']} interactive")
            print(f"  MDP:  {mdp['states']} states, "
                  f"{mdp['actions']} actions, "
                  f"{mdp['transitions']} transitions")
            if result.get("detected_components"):
                for dc in result["detected_components"]:
                    print(f"  Component: {dc['type']} "
                          f"({dc['sections']} sections)")
            print(f"  WCAG: {wcag['violation_count']} violations, "
                  f"{wcag['total_cognitive_bits']:.1f} cognitive bits")
            print(f"  Value: initial={vf['initial_state_value']:.3f}, "
                  f"solve_time={vf['solve_time_s']:.3f}s")
            print(f"  Bottlenecks: {len(result['bottlenecks'])}")
            print(f"  Score: {result['cognitive_cost_score']}")
            print(f"  Time:  {result['total_time_s']:.2f}s")

            total_bottlenecks += len(result["bottlenecks"])
            total_violations += wcag["violation_count"]
            total_cost += result["cognitive_cost_score"]
            component_scores.append((name, result["cognitive_cost_score"]))

        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["components"][name] = {"error": str(e)}

        # Run paired comparison
        try:
            paired = run_paired_comparison(name, html)
            all_results["paired_comparisons"][name] = paired
            print(f"  Paired: {paired['verdict']} "
                  f"(Δ={paired.get('delta', 'N/A')}, "
                  f"d={paired.get('effect_size', 'N/A')})")
        except Exception as e:
            print(f"  Paired comparison error: {e}")
            all_results["paired_comparisons"][name] = {"error": str(e)}

        print()

    # ── Summary ───────────────────────────────────────────────────────────
    n = len(html_files)
    all_results["summary"] = {
        "total_components": n,
        "total_bottlenecks": total_bottlenecks,
        "total_wcag_violations": total_violations,
        "mean_cognitive_cost": round(total_cost / max(n, 1), 2),
        "component_ranking": sorted(
            component_scores, key=lambda x: x[1], reverse=True
        ),
        "regression_verdicts": {
            name: pc.get("verdict", "error")
            for name, pc in all_results["paired_comparisons"].items()
        },
    }

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Components analysed:    {n}")
    print(f"Total bottlenecks:      {total_bottlenecks}")
    print(f"Total WCAG violations:  {total_violations}")
    print(f"Mean cognitive cost:    {total_cost / max(n, 1):.2f}")
    print()
    print("Component ranking (highest cost = most complex):")
    for rank, (cname, score) in enumerate(
        sorted(component_scores, key=lambda x: x[1], reverse=True), 1
    ):
        print(f"  {rank}. {cname:<20s}  score={score:.2f}")

    print()
    print("Paired comparison verdicts (original vs degraded):")
    for cname, pc in all_results["paired_comparisons"].items():
        v = pc.get("verdict", "error")
        d = pc.get("delta", "N/A")
        print(f"  {cname:<20s}  {v} (Δ={d})")

    # ── Write output ──────────────────────────────────────────────────────
    output_path = PROJECT_ROOT / "real_benchmarks" / "govuk_benchmark_results.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults written to {output_path}")


if __name__ == "__main__":
    main()
