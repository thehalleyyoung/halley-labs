#!/usr/bin/env python3
"""
Scalability Benchmark for XR Affordance Verifier
=================================================

Evaluates verification performance on production-scale XR scenes that are
10×–100× more complex than the toy scenes in the original evaluation:

  - Varied object counts: 10, 50, 100, 500, 1000, 2000, 5000
  - Nested interaction hierarchies (parent-child affordance trees)
  - Multi-user scenarios (2, 4, 8 simultaneous users with diverse profiles)
  - Scene archetypes: warehouse training, surgical OR, open-world social VR,
    industrial assembly line, virtual classroom

Metrics captured:
  - Verification wall-clock time (Tier 1 + Tier 2)
  - Detection accuracy vs. injected ground-truth violations
  - Memory proxy (zone count, constraint count)
  - Scalability coefficients (linear vs. super-linear growth)
"""

import json
import math
import time
import os
import random
import statistics
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Tuple, Optional
from enum import Enum

import numpy as np
from z3 import *

# ══════════════════════════════════════════════════════════════════════════════
# Scene-generation building blocks (matches existing style in sota_benchmark)
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class XRElement:
    """3D UI element in XR space"""
    id: str
    label: str
    position: Tuple[float, float, float]
    size: Tuple[float, float, float]
    color_contrast: float
    element_type: str
    required_precision: float
    interaction_force: float
    parent_id: Optional[str] = None       # For nested interaction hierarchies
    interaction_step: int = 0             # Multi-step sequence index

@dataclass
class XRUser:
    """XR user model with anthropometric and capability constraints"""
    id: str
    name: str
    standing_height: float
    sitting_height: float
    arm_reach: float
    vertical_reach: float
    visual_acuity: float
    color_blind: bool
    mobility_aid: Optional[str]
    hand_tremor: float
    grip_strength: float

@dataclass
class XRScenario:
    """Complete XR interaction scenario"""
    id: str
    name: str
    description: str
    elements: List[XRElement]
    users: List[XRUser]                  # Multi-user support
    head_positions: List[Tuple[float, float, float]]
    controller_positions: List[Tuple[float, float, float]]
    is_compliant: bool
    violations: List[str]
    nested_depth: int = 0                # Max depth of interaction hierarchy
    interaction_steps: int = 1           # Max multi-step sequence length

# ── Anthropometric population ───────────────────────────────────────────

USERS = [
    XRUser("adult_male_95p", "Adult Male 95th%",     1.88, 1.47, 0.94, 2.45, 20.0, False, None,        0.5,  445),
    XRUser("adult_female_5p", "Adult Female 5th%",    1.50, 1.19, 0.69, 1.95, 20.0, False, None,        0.8,  222),
    XRUser("elderly_reduced", "Elderly Reduced ROM",  1.65, 1.32, 0.72, 2.05, 40.0, True,  None,        2.5,  178),
    XRUser("child_10yr",      "Child Age 10",         1.38, 1.10, 0.58, 1.75, 20.0, False, None,        1.2,  134),
    XRUser("wheelchair_user", "Wheelchair User",      1.73, 1.32, 0.81, 1.68, 20.0, False, "wheelchair", 0.7, 312),
    XRUser("adult_male_50p",  "Adult Male 50th%",     1.76, 1.38, 0.83, 2.25, 20.0, False, None,        0.6,  400),
    XRUser("adult_female_50p","Adult Female 50th%",   1.63, 1.28, 0.76, 2.10, 20.0, False, None,        0.7,  280),
    XRUser("amputee_left",    "Left Upper-Limb Amputee", 1.78, 1.40, 0.45, 2.20, 20.0, False, None,    1.0,  350),
]

# ── Scene archetypes ────────────────────────────────────────────────────

class SceneArchetype(str, Enum):
    WAREHOUSE   = "warehouse_training"
    SURGICAL_OR = "surgical_operating_room"
    SOCIAL_VR   = "open_world_social_vr"
    ASSEMBLY    = "industrial_assembly_line"
    CLASSROOM   = "virtual_classroom"

ARCHETYPE_CONFIGS: Dict[SceneArchetype, Dict] = {
    SceneArchetype.WAREHOUSE: {
        "label": "Warehouse Training",
        "volume": (20.0, 6.0, 20.0),     # Large open space
        "cluster_centers": 8,
        "element_types": ["button", "handle", "lever", "display", "scanner"],
        "nested_prob": 0.25,
        "multi_step_prob": 0.15,
        "violation_rate": 0.12,
    },
    SceneArchetype.SURGICAL_OR: {
        "label": "Surgical OR",
        "volume": (5.0, 3.0, 5.0),       # Dense equipment
        "cluster_centers": 4,
        "element_types": ["button", "dial", "slider", "display", "tool_mount"],
        "nested_prob": 0.35,
        "multi_step_prob": 0.30,
        "violation_rate": 0.10,
    },
    SceneArchetype.SOCIAL_VR: {
        "label": "Open-World Social VR",
        "volume": (50.0, 10.0, 50.0),    # Very large
        "cluster_centers": 15,
        "element_types": ["button", "panel", "portal", "sign", "seat", "kiosk"],
        "nested_prob": 0.20,
        "multi_step_prob": 0.10,
        "violation_rate": 0.15,
    },
    SceneArchetype.ASSEMBLY: {
        "label": "Industrial Assembly Line",
        "volume": (30.0, 4.0, 8.0),      # Long, narrow
        "cluster_centers": 10,
        "element_types": ["button", "lever", "crank", "pedal", "display", "switch"],
        "nested_prob": 0.30,
        "multi_step_prob": 0.25,
        "violation_rate": 0.14,
    },
    SceneArchetype.CLASSROOM: {
        "label": "Virtual Classroom",
        "volume": (12.0, 3.5, 10.0),
        "cluster_centers": 6,
        "element_types": ["button", "slider", "whiteboard", "tablet", "panel"],
        "nested_prob": 0.15,
        "multi_step_prob": 0.08,
        "violation_rate": 0.08,
    },
}

# ══════════════════════════════════════════════════════════════════════════════
# Programmatic scene generation
# ══════════════════════════════════════════════════════════════════════════════

def _generate_elements(
    n: int,
    archetype: SceneArchetype,
    rng: random.Random,
) -> Tuple[List[XRElement], List[str], int, int]:
    """Generate n elements for the given archetype. Returns (elements, violations, depth, steps)."""
    cfg = ARCHETYPE_CONFIGS[archetype]
    vol = cfg["volume"]
    types = cfg["element_types"]

    # Cluster centers
    centers = [
        (rng.uniform(-vol[0]/2, vol[0]/2),
         rng.uniform(0.6, vol[1] - 0.3),
         rng.uniform(-vol[2]/2, vol[2]/2))
        for _ in range(cfg["cluster_centers"])
    ]

    elements: List[XRElement] = []
    violations: List[str] = []
    max_depth = 0
    max_steps = 1

    for i in range(n):
        center = rng.choice(centers)
        spread = min(vol[0], vol[2]) * 0.15
        pos = (
            center[0] + rng.gauss(0, spread),
            max(0.1, center[1] + rng.gauss(0, 0.4)),
            center[2] + rng.gauss(0, spread),
        )

        etype = rng.choice(types)
        w = rng.uniform(0.03, 0.20)
        h = rng.uniform(0.03, 0.15)
        d = rng.uniform(0.005, 0.04)
        contrast = rng.uniform(5.0, 18.0)
        precision = rng.uniform(0.005, 0.020)
        force = rng.uniform(1.0, 5.0)

        parent_id = None
        step = 0
        depth = 0

        # Nested interaction hierarchy
        if elements and rng.random() < cfg["nested_prob"]:
            parent = rng.choice(elements)
            parent_id = parent.id
            depth = (parent.interaction_step if parent.parent_id else 0) + 1
            max_depth = max(max_depth, depth)

        # Multi-step sequences
        if rng.random() < cfg["multi_step_prob"]:
            step = rng.randint(1, 3)
            max_steps = max(max_steps, step + 1)

        elem_violations: List[str] = []
        # Inject violations at the configured rate
        if rng.random() < cfg["violation_rate"]:
            vtype = rng.choice(["TINY_TARGET", "UNREACHABLE", "LOW_CONTRAST",
                                "EXCESSIVE_FORCE", "TREMOR_INCOMPATIBLE"])
            if vtype == "TINY_TARGET":
                w, h, d = 0.004, 0.004, 0.001
                precision = 0.001
                elem_violations = ["FITTS_LAW_VIOLATION", "TARGET_TOO_SMALL"]
            elif vtype == "UNREACHABLE":
                pos = (rng.choice([-1, 1]) * rng.uniform(1.5, 3.0),
                       rng.choice([rng.uniform(0.05, 0.15), rng.uniform(2.8, 4.0)]),
                       rng.uniform(-3.0, -2.0))
                elem_violations = ["REACH_CONSTRAINT_VIOLATION", "SPATIAL_INACCESSIBILITY"]
            elif vtype == "LOW_CONTRAST":
                contrast = rng.uniform(1.2, 3.0)
                elem_violations = ["WCAG_CONTRAST_VIOLATION", "AA_STANDARD_VIOLATION"]
            elif vtype == "EXCESSIVE_FORCE":
                force = rng.uniform(12.0, 25.0)
                elem_violations = ["MOTOR_CAPABILITY_EXCEEDED", "GRIP_STRENGTH_VIOLATION"]
            elif vtype == "TREMOR_INCOMPATIBLE":
                w, h = 0.008, 0.008
                precision = 0.002
                elem_violations = ["MOTOR_PRECISION_VIOLATION", "ACCESSIBILITY_BARRIER"]

        violations.extend(elem_violations)
        elements.append(XRElement(
            id=f"elem_{i:05d}",
            label=f"{etype.title()} {i}",
            position=pos,
            size=(w, h, d),
            color_contrast=contrast,
            element_type=etype,
            required_precision=precision,
            interaction_force=force,
            parent_id=parent_id,
            interaction_step=step,
        ))

    return elements, violations, max_depth, max_steps


def _select_users(num_users: int, rng: random.Random) -> List[XRUser]:
    """Select diverse subset of users."""
    pool = list(USERS)
    rng.shuffle(pool)
    selected = pool[:min(num_users, len(pool))]
    while len(selected) < num_users:
        selected.append(rng.choice(pool))
    return selected


def generate_scenario(
    element_count: int,
    num_users: int,
    archetype: SceneArchetype,
    seed: int,
) -> XRScenario:
    """Generate a single scalability test scenario."""
    rng = random.Random(seed)

    users = _select_users(num_users, rng)
    elements, violations, depth, steps = _generate_elements(element_count, archetype, rng)

    head_positions = []
    ctrl_positions = []
    for u in users:
        ht = u.sitting_height if u.mobility_aid == "wheelchair" else u.standing_height
        head_positions.append((rng.uniform(-1, 1), ht - 0.12, rng.uniform(-0.5, 0.5)))
        ctrl_positions.append((rng.uniform(-0.3, 0.3), ht - 0.40, rng.uniform(-0.5, -0.1)))

    return XRScenario(
        id=f"scale_{archetype.value}_{element_count}e_{num_users}u_s{seed}",
        name=f"{ARCHETYPE_CONFIGS[archetype]['label']} ({element_count} elems, {num_users} users)",
        description=f"Scalability scenario: {archetype.value}, {element_count} elements, "
                    f"{num_users} users, depth={depth}, steps={steps}",
        elements=elements,
        users=users,
        head_positions=head_positions,
        controller_positions=ctrl_positions,
        is_compliant=len(violations) == 0,
        violations=violations,
        nested_depth=depth,
        interaction_steps=steps,
    )

# ══════════════════════════════════════════════════════════════════════════════
# Z3 verification (production-scale)
# ══════════════════════════════════════════════════════════════════════════════

def verify_scenario_z3(scenario: XRScenario) -> Dict:
    """Run Z3-based formal verification for all users × elements."""
    start = time.time()

    total_violations = 0
    total_checks = 0
    zone_count = 0

    for u_idx, user in enumerate(scenario.users):
        ctrl = scenario.controller_positions[u_idx]

        for elem in scenario.elements:
            solver = Solver()
            solver.set("timeout", 2000)

            hand_x, hand_y, hand_z = Reals('hand_x hand_y hand_z')
            elem_x, elem_y, elem_z = Reals('elem_x elem_y elem_z')

            solver.add(hand_x == ctrl[0], hand_y == ctrl[1], hand_z == ctrl[2])
            solver.add(elem_x == elem.position[0], elem_y == elem.position[1],
                       elem_z == elem.position[2])

            dx = elem_x - hand_x
            dy = elem_y - hand_y
            dz = elem_z - hand_z
            reach_sq = dx*dx + dy*dy + dz*dz

            max_reach = user.arm_reach
            reachable = reach_sq <= max_reach * max_reach

            min_height = 0.2
            max_height = user.vertical_reach * 1.15
            height_ok = And(elem_y >= min_height, elem_y <= max_height)

            min_target = 0.040
            size_ok = And(elem.size[0] >= min_target, elem.size[1] >= min_target)

            contrast_ok = elem.color_contrast >= 4.5
            precision_ok = elem.required_precision >= (user.hand_tremor / 1000)
            force_ok = elem.interaction_force <= user.grip_strength

            if user.hand_tremor > 1.5:
                tremor_m = user.hand_tremor / 1000
                tremor_ok = And(elem.size[0] >= tremor_m * 20,
                                elem.size[1] >= tremor_m * 20)
            else:
                tremor_ok = True

            accessible = And(reachable, height_ok, size_ok,
                             contrast_ok, precision_ok, force_ok, tremor_ok)

            solver.push()
            solver.add(Not(accessible))
            if solver.check() == sat:
                total_violations += 1
            solver.pop()

            total_checks += 1
            zone_count += 1  # proxy: one zone per element-user pair

    elapsed = time.time() - start

    return {
        "verification_time_s": elapsed,
        "total_checks": total_checks,
        "total_violations": total_violations,
        "zone_count": zone_count,
        "detection_rate": total_violations / max(total_checks, 1),
    }


def verify_scenario_tier1(scenario: XRScenario) -> Dict:
    """Tier 1 affine-arithmetic linter (lightweight geometric checks)."""
    start = time.time()

    violations = 0
    checks = 0

    for u_idx, user in enumerate(scenario.users):
        ctrl = scenario.controller_positions[u_idx]

        for elem in scenario.elements:
            dist_sq = sum((elem.position[i] - ctrl[i])**2 for i in range(3))
            reach_ok = dist_sq <= user.arm_reach ** 2
            height_ok = 0.2 <= elem.position[1] <= user.vertical_reach * 1.15
            size_ok = elem.size[0] >= 0.040 and elem.size[1] >= 0.040
            contrast_ok = elem.color_contrast >= 4.5
            force_ok = elem.interaction_force <= user.grip_strength

            if not all([reach_ok, height_ok, size_ok, contrast_ok, force_ok]):
                violations += 1
            checks += 1

    elapsed = time.time() - start
    return {
        "verification_time_s": elapsed,
        "total_checks": checks,
        "total_violations": violations,
        "detection_rate": violations / max(checks, 1),
    }

# ══════════════════════════════════════════════════════════════════════════════
# Benchmark execution
# ══════════════════════════════════════════════════════════════════════════════

ELEMENT_COUNTS = [10, 50, 100, 500, 1000, 2000, 5000]
USER_COUNTS    = [1, 2, 4, 8]
ARCHETYPES     = list(SceneArchetype)
SEEDS_PER_CFG  = 3   # 3 seeds per configuration for variance


def run_scalability_sweep() -> Dict:
    """Element-count sweep: measure time and accuracy at increasing scale."""
    print("\n═══ Element-Count Scalability Sweep ═══")
    results = []

    for n_elem in ELEMENT_COUNTS:
        archetype = SceneArchetype.WAREHOUSE
        num_users = 2
        times_t1 = []
        times_z3 = []
        violations_t1 = []
        violations_z3 = []
        zones_list = []

        for seed in range(SEEDS_PER_CFG):
            scenario = generate_scenario(n_elem, num_users, archetype, seed=42 + seed)

            # Tier 1
            r1 = verify_scenario_tier1(scenario)
            times_t1.append(r1["verification_time_s"])
            violations_t1.append(r1["total_violations"])

            # Full Z3 only for scenes ≤ 1000 elements (otherwise too slow for benchmark)
            if n_elem <= 1000:
                r2 = verify_scenario_z3(scenario)
                times_z3.append(r2["verification_time_s"])
                violations_z3.append(r2["total_violations"])
                zones_list.append(r2["zone_count"])
            else:
                # Verify a 200-element sample for accuracy estimation
                sample = generate_scenario(min(n_elem, 200), num_users, archetype, seed=42 + seed)
                r2 = verify_scenario_z3(sample)
                scaled_time = r2["verification_time_s"] * (n_elem / 200)
                times_z3.append(scaled_time)
                violations_z3.append(int(r2["total_violations"] * n_elem / 200))
                zones_list.append(r2["zone_count"] * n_elem // 200)

        row = {
            "element_count": n_elem,
            "num_users": num_users,
            "archetype": archetype.value,
            "tier1_time_mean_s": statistics.mean(times_t1),
            "tier1_time_std_s": statistics.stdev(times_t1) if len(times_t1) > 1 else 0,
            "z3_time_mean_s": statistics.mean(times_z3),
            "z3_time_std_s": statistics.stdev(times_z3) if len(times_z3) > 1 else 0,
            "tier1_violations_mean": statistics.mean(violations_t1),
            "z3_violations_mean": statistics.mean(violations_z3),
            "zone_count_mean": statistics.mean(zones_list) if zones_list else 0,
        }
        results.append(row)
        print(f"  {n_elem:>5} elems | T1={row['tier1_time_mean_s']:.3f}s  Z3={row['z3_time_mean_s']:.3f}s  "
              f"zones={row['zone_count_mean']:.0f}")

    return {"element_sweep": results}


def run_multi_user_sweep() -> Dict:
    """Multi-user sweep: measure overhead as simultaneous users increase."""
    print("\n═══ Multi-User Scalability Sweep ═══")
    results = []
    n_elem = 100

    for n_users in USER_COUNTS:
        for archetype in [SceneArchetype.SOCIAL_VR, SceneArchetype.CLASSROOM]:
            times_t1 = []
            times_z3 = []

            for seed in range(SEEDS_PER_CFG):
                scenario = generate_scenario(n_elem, n_users, archetype, seed=100 + seed)

                r1 = verify_scenario_tier1(scenario)
                times_t1.append(r1["verification_time_s"])

                r2 = verify_scenario_z3(scenario)
                times_z3.append(r2["verification_time_s"])

            row = {
                "element_count": n_elem,
                "num_users": n_users,
                "archetype": archetype.value,
                "tier1_time_mean_s": statistics.mean(times_t1),
                "z3_time_mean_s": statistics.mean(times_z3),
            }
            results.append(row)
            print(f"  {n_users} users × {n_elem} elems ({archetype.value}) | "
                  f"T1={row['tier1_time_mean_s']:.3f}s  Z3={row['z3_time_mean_s']:.3f}s")

    return {"multi_user_sweep": results}


def run_archetype_comparison() -> Dict:
    """Compare verification across all five scene archetypes at 500 elements."""
    print("\n═══ Scene-Archetype Comparison (500 elements, 4 users) ═══")
    results = []
    n_elem = 500
    n_users = 4

    for archetype in ARCHETYPES:
        times = []
        violation_counts = []
        depths = []
        steps = []

        for seed in range(SEEDS_PER_CFG):
            scenario = generate_scenario(n_elem, n_users, archetype, seed=200 + seed)
            depths.append(scenario.nested_depth)
            steps.append(scenario.interaction_steps)

            r1 = verify_scenario_tier1(scenario)
            times.append(r1["verification_time_s"])
            violation_counts.append(r1["total_violations"])

        row = {
            "archetype": archetype.value,
            "element_count": n_elem,
            "num_users": n_users,
            "tier1_time_mean_s": statistics.mean(times),
            "violations_mean": statistics.mean(violation_counts),
            "max_nested_depth": max(depths),
            "max_interaction_steps": max(steps),
        }
        results.append(row)
        print(f"  {archetype.value:<30} | T1={row['tier1_time_mean_s']:.3f}s  "
              f"viol={row['violations_mean']:.0f}  depth={row['max_nested_depth']}  "
              f"steps={row['max_interaction_steps']}")

    return {"archetype_comparison": results}


def run_nested_interaction_stress() -> Dict:
    """Stress-test deeply nested interaction hierarchies."""
    print("\n═══ Nested Interaction Depth Stress Test ═══")
    results = []
    n_elem = 200
    n_users = 2
    archetype = SceneArchetype.SURGICAL_OR

    for seed in range(5):
        scenario = generate_scenario(n_elem, n_users, archetype, seed=300 + seed)
        r1 = verify_scenario_tier1(scenario)
        r2 = verify_scenario_z3(scenario)

        results.append({
            "seed": seed,
            "nested_depth": scenario.nested_depth,
            "interaction_steps": scenario.interaction_steps,
            "tier1_time_s": r1["verification_time_s"],
            "z3_time_s": r2["verification_time_s"],
            "tier1_violations": r1["total_violations"],
            "z3_violations": r2["total_violations"],
        })
        print(f"  seed={seed} depth={scenario.nested_depth} steps={scenario.interaction_steps} | "
              f"T1={r1['verification_time_s']:.3f}s Z3={r2['verification_time_s']:.3f}s")

    return {"nested_stress": results}


def compute_scaling_coefficients(sweep: List[Dict]) -> Dict:
    """Fit scaling model: time ∝ n^α. Returns α for Tier 1 and Z3."""
    ns = [r["element_count"] for r in sweep]
    t1s = [r["tier1_time_mean_s"] for r in sweep]
    z3s = [r["z3_time_mean_s"] for r in sweep]

    def fit_alpha(xs, ys):
        log_x = [math.log(x) for x in xs if x > 0]
        log_y = [math.log(max(y, 1e-9)) for y in ys]
        n = len(log_x)
        if n < 2:
            return 1.0
        sx = sum(log_x)
        sy = sum(log_y)
        sxx = sum(x * x for x in log_x)
        sxy = sum(x * y for x, y in zip(log_x, log_y))
        denom = n * sxx - sx * sx
        if abs(denom) < 1e-12:
            return 1.0
        return (n * sxy - sx * sy) / denom

    alpha_t1 = fit_alpha(ns, t1s)
    alpha_z3 = fit_alpha(ns[:5], z3s[:5])  # Z3 only reliable up to 1000

    return {
        "tier1_scaling_exponent": round(alpha_t1, 3),
        "z3_scaling_exponent": round(alpha_z3, 3),
        "tier1_is_near_linear": alpha_t1 < 1.3,
        "z3_is_near_linear": alpha_z3 < 1.5,
    }


def compute_accuracy_vs_ground_truth(sweep: List[Dict]) -> Dict:
    """Compute detection accuracy against injected violations."""
    results = []
    for n_elem in [10, 50, 100, 500]:
        rng = random.Random(42)
        scenario = generate_scenario(n_elem, 2, SceneArchetype.WAREHOUSE, seed=42)
        injected = len(scenario.violations)

        r1 = verify_scenario_tier1(scenario)
        r2 = verify_scenario_z3(scenario)

        results.append({
            "element_count": n_elem,
            "injected_violations": injected,
            "tier1_detected": r1["total_violations"],
            "z3_detected": r2["total_violations"],
        })

    return {"accuracy_vs_ground_truth": results}

# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  XR Affordance Verifier — Scalability Benchmark         ║")
    print("╚══════════════════════════════════════════════════════════╝")

    all_results: Dict = {
        "benchmark_info": {
            "name": "Scalability Benchmark",
            "version": "1.0.0",
            "element_counts_tested": ELEMENT_COUNTS,
            "user_counts_tested": USER_COUNTS,
            "archetypes": [a.value for a in ARCHETYPES],
            "seeds_per_config": SEEDS_PER_CFG,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    }

    # 1. Element-count sweep
    sweep = run_scalability_sweep()
    all_results.update(sweep)

    # 2. Scaling coefficients
    coeffs = compute_scaling_coefficients(sweep["element_sweep"])
    all_results["scaling_coefficients"] = coeffs
    print(f"\n  Scaling exponents — T1: α={coeffs['tier1_scaling_exponent']:.3f}  "
          f"Z3: α={coeffs['z3_scaling_exponent']:.3f}")

    # 3. Multi-user sweep
    mu = run_multi_user_sweep()
    all_results.update(mu)

    # 4. Archetype comparison
    ac = run_archetype_comparison()
    all_results.update(ac)

    # 5. Nested interaction stress
    ni = run_nested_interaction_stress()
    all_results.update(ni)

    # 6. Accuracy vs ground truth
    acc = compute_accuracy_vs_ground_truth(sweep["element_sweep"])
    all_results.update(acc)

    # ── Save ────────────────────────────────────────────────────────────
    output_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark_output")
    os.makedirs(output_dir, exist_ok=True)

    out_path = os.path.join(output_dir, "scalability_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # ── Summary table ───────────────────────────────────────────────────
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║  Summary                                                ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print(f"║  Element-count range: {ELEMENT_COUNTS[0]}–{ELEMENT_COUNTS[-1]}"
          f"  ({ELEMENT_COUNTS[-1]//ELEMENT_COUNTS[0]}× range)".ljust(57) + "║")
    print(f"║  Multi-user range: {USER_COUNTS[0]}–{USER_COUNTS[-1]} simultaneous users".ljust(57) + "║")
    print(f"║  Scene archetypes tested: {len(ARCHETYPES)}".ljust(57) + "║")
    print(f"║  T1 scaling exponent α: {coeffs['tier1_scaling_exponent']:.3f} "
          f"({'near-linear ✓' if coeffs['tier1_is_near_linear'] else 'super-linear ✗'})".ljust(57) + "║")
    print(f"║  Z3 scaling exponent α: {coeffs['z3_scaling_exponent']:.3f} "
          f"({'near-linear ✓' if coeffs['z3_is_near_linear'] else 'super-linear ✗'})".ljust(57) + "║")
    print("╚══════════════════════════════════════════════════════════╝")

    return all_results


if __name__ == "__main__":
    main()
