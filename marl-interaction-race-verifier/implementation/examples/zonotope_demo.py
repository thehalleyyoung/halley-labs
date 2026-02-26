"""Zonotope abstract domain demonstration.

Demonstrates the zonotope abstract domain used in MARACE for
reachability analysis: construction, affine transforms, join,
widening, halfspace intersection, generator management, fixpoint
iteration, and 2-D projection for visualization.

Usage::

    python -m examples.zonotope_demo
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from marace.abstract.zonotope import Zonotope
from marace.abstract.fixpoint import (
    ConvergenceChecker,
    FixpointEngine,
    FixpointResult,
    IterationState,
    WideningStrategy,
)
from marace.abstract.hb_constraints import HBConstraint, HBConstraintSet


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'─' * 60}")
    print(f"  {title}")
    print(f"{'─' * 60}")


# ── 1. Construction ──────────────────────────────────────────────────

def demo_construction() -> None:
    """Show different ways to construct zonotopes."""
    section("1. Zonotope Construction")

    # From interval box
    z_box = Zonotope.from_interval(
        lower=np.array([-1.0, -2.0, 0.0]),
        upper=np.array([3.0, 1.0, 4.0]),
    )
    print(f"From interval:  center={z_box.center}, dim={z_box.dimension}, "
          f"gens={z_box.num_generators}")
    lo, hi = z_box.bounding_box
    print(f"  Bounding box: [{lo}] to [{hi}]")

    # From a single point
    z_pt = Zonotope.from_point(np.array([1.0, 2.0]))
    print(f"\nFrom point:     center={z_pt.center}, gens={z_pt.num_generators}")

    # Unit ball
    z_unit = Zonotope.unit_ball(3, num_generators=5)
    print(f"\nUnit ball ℓ∞:   center={z_unit.center}, "
          f"dim={z_unit.dimension}, gens={z_unit.num_generators}")

    # Direct construction with explicit generators
    G = np.array([[1.0, 0.5], [0.0, 1.0]])
    z_custom = Zonotope(center=np.array([0.0, 0.0]), generators=G)
    print(f"\nCustom:         center={z_custom.center}")
    print(f"  Generators:\n{z_custom.generators}")
    print(f"  Volume bound: {z_custom.volume_bound:.4f}")


# ── 2. Core operations ──────────────────────────────────────────────

def demo_operations() -> None:
    """Demonstrate affine transform, join, and Minkowski sum."""
    section("2. Core Operations")

    z = Zonotope.from_interval(np.array([-1.0, -1.0]), np.array([1.0, 1.0]))

    # Affine transform: rotation by 45° + scaling
    theta = np.pi / 4
    W = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
    ]) * 1.5
    b = np.array([2.0, 3.0])

    z_affine = z.affine_transform(W, b)
    print(f"Original:  center={z.center}, gens={z.num_generators}")
    print(f"Affine:    center={z_affine.center.round(3)}, gens={z_affine.num_generators}")
    lo, hi = z_affine.bounding_box
    print(f"  Bbox: [{lo.round(3)}] to [{hi.round(3)}]")

    # Join (convex hull over-approximation)
    z1 = Zonotope.from_interval(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    z2 = Zonotope.from_interval(np.array([3.0, 1.0]), np.array([5.0, 3.0]))
    z_join = z1.join(z2)
    print(f"\nJoin:")
    print(f"  Z1 bbox: {z1.bounding_box}")
    print(f"  Z2 bbox: {z2.bounding_box}")
    print(f"  Join bbox: center={z_join.center}, gens={z_join.num_generators}")
    lo, hi = z_join.bounding_box
    print(f"  [{lo}] to [{hi}]")

    # Minkowski sum
    noise = Zonotope.from_interval(np.array([-0.1, -0.1]), np.array([0.1, 0.1]))
    z_sum = z1.minkowski_sum(noise)
    print(f"\nMinkowski sum:")
    print(f"  Z1 + noise: gens {z1.num_generators} + {noise.num_generators} = "
          f"{z_sum.num_generators}")


# ── 3. Halfspace intersection (HB-constraint pruning) ────────────────

def demo_hb_pruning() -> None:
    """Demonstrate zonotope pruning with HB-derived constraints."""
    section("3. HB-Constraint Pruning")

    z = Zonotope.from_interval(np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
    print(f"Before pruning: bbox = {z.bounding_box}")
    print(f"  Volume bound: {z.volume_bound:.2f}")

    # Constraint: x₁ - x₂ ≤ 2 (from HB ordering: agent 1's time ≤ agent 2's + 2)
    a1 = np.array([1.0, -1.0])
    z_pruned = z.meet_halfspace(a1, 2.0)
    lo, hi = z_pruned.bounding_box
    print(f"\nAfter x₁ - x₂ ≤ 2:")
    print(f"  Bbox: [{lo.round(3)}] to [{hi.round(3)}]")
    print(f"  Volume bound: {z_pruned.volume_bound:.2f}")

    # Second constraint: x₂ ≤ 3
    a2 = np.array([0.0, 1.0])
    z_pruned2 = z_pruned.meet_halfspace(a2, 3.0)
    lo2, hi2 = z_pruned2.bounding_box
    print(f"\nAfter also x₂ ≤ 3:")
    print(f"  Bbox: [{lo2.round(3)}] to [{hi2.round(3)}]")
    print(f"  Volume bound: {z_pruned2.volume_bound:.2f}")

    # Point containment check
    inside = z_pruned2.contains_point(np.array([0.0, 0.0]))
    outside = z_pruned2.contains_point(np.array([10.0, 10.0]))
    print(f"\n  (0,0) ∈ Z_pruned: {inside}")
    print(f"  (10,10) ∈ Z_pruned: {outside}")


# ── 4. Widening ──────────────────────────────────────────────────────

def demo_widening() -> None:
    """Demonstrate widening for fixpoint convergence guarantee."""
    section("4. Widening Operator")

    z_old = Zonotope.from_interval(np.array([0.0, 0.0]), np.array([2.0, 2.0]))
    z_new = Zonotope.from_interval(np.array([-0.5, 0.0]), np.array([2.5, 2.3]))

    z_widened = z_old.widening(z_new, threshold=1.05)

    print(f"Old:     bbox = {[b.round(3) for b in z_old.bounding_box]}")
    print(f"New:     bbox = {[b.round(3) for b in z_new.bounding_box]}")
    print(f"Widened: bbox = {[b.round(3) for b in z_widened.bounding_box]}")
    print(f"\nWidening ensures convergence by extrapolating growth directions.")


# ── 5. Generator management ─────────────────────────────────────────

def demo_generator_management() -> None:
    """Show generator reduction and merging."""
    section("5. Generator Management")

    # Create zonotope with many generators
    rng = np.random.default_rng(42)
    center = np.zeros(3)
    generators = rng.normal(0, 1, size=(3, 20))
    z = Zonotope(center=center, generators=generators)

    print(f"Original: {z.num_generators} generators")
    lo, hi = z.bounding_box
    print(f"  Bbox width: {(hi - lo).round(3)}")

    # Reduce to 8 generators (Girard PCA method)
    z_reduced = z.reduce_generators(8)
    print(f"\nReduced to {z_reduced.num_generators} generators (Girard PCA)")
    lo_r, hi_r = z_reduced.bounding_box
    print(f"  Bbox width: {(hi_r - lo_r).round(3)}")
    print(f"  Sound: reduced zonotope ⊇ original (by construction)")

    # Remove near-zero generators
    z_clean = z.remove_zero_generators(tol=0.1)
    print(f"\nAfter removing ‖g‖ < 0.1: {z_clean.num_generators} generators")


# ── 6. Fixpoint iteration convergence ────────────────────────────────

def demo_fixpoint() -> None:
    """Demonstrate fixpoint iteration with convergence tracking."""
    section("6. Fixpoint Iteration")

    dim = 4
    z0 = Zonotope.from_interval(np.full(dim, -1.0), np.full(dim, 1.0))

    # Contracting linear map (eigenvalues < 1) + small noise
    A = 0.9 * np.eye(dim)
    A[0, 1] = 0.1
    A[1, 0] = -0.1
    noise_half = np.full(dim, 0.05)

    # Add HB constraint: x₀ + x₁ ≤ 3
    constraints = HBConstraintSet()
    normal = np.zeros(dim)
    normal[0] = 1.0
    normal[1] = 1.0
    constraints.add(HBConstraint(
        normal=normal, bound=3.0,
        label="timing ordering constraint",
    ))

    def transfer(z: Zonotope) -> Zonotope:
        z_next = z.affine_transform(A)
        noise = Zonotope.from_interval(-noise_half, noise_half)
        return z_next.minkowski_sum(noise)

    engine = FixpointEngine(
        transfer_fn=transfer,
        strategy=WideningStrategy.DELAYED,
        max_iterations=30,
        convergence_threshold=1e-5,
        convergence_patience=3,
        delay_widening=5,
        max_generators=20,
        hb_constraints=constraints,
    )

    result = engine.compute(z0)

    print(f"Converged: {result.converged}")
    print(f"Iterations: {result.iterations}")
    print(f"Wall time: {result.wall_time_s:.4f}s")
    print(f"HB-consistent: {result.hb_consistent}")
    print(f"Final dimension: {result.element.dimension}, "
          f"generators: {result.element.num_generators}")
    lo, hi = result.element.bounding_box
    print(f"Final bbox: [{lo.round(4)}] to [{hi.round(4)}]")

    # Print convergence history
    print("\nIteration history:")
    print(f"  {'Iter':>4}  {'Hausdorff':>10}  {'Volume':>12}  {'Gens':>5}  {'HB':>4}")
    for s in result.iteration_history[:10]:
        print(f"  {s.iteration:4d}  {s.hausdorff_from_prev:10.6f}  "
              f"{s.volume_bound:12.4f}  {s.num_generators:5d}  "
              f"{'✓' if s.hb_consistent else '✗':>4}")
    if result.iterations > 10:
        print(f"  ... ({result.iterations - 10} more iterations)")


# ── 7. 2-D projection visualization ──────────────────────────────────

def demo_2d_visualization() -> None:
    """Show 2-D projection vertices for plotting."""
    section("7. 2-D Projection (ASCII Visualization)")

    z = Zonotope(
        center=np.array([0.0, 0.0]),
        generators=np.array([
            [1.0, 0.5, 0.3],
            [0.0, 0.8, -0.4],
        ]),
    )
    verts = z.vertices_2d()
    print(f"Zonotope: center={z.center}, {z.num_generators} generators")
    print(f"2-D vertices ({len(verts)} points):")
    for i, v in enumerate(verts):
        print(f"  v{i}: ({v[0]:+.3f}, {v[1]:+.3f})")

    # ASCII plot
    lo, hi = z.bounding_box
    grid_w, grid_h = 50, 20
    x_range = (lo[0] - 0.5, hi[0] + 0.5)
    y_range = (lo[1] - 0.5, hi[1] + 0.5)

    canvas = [[' ' for _ in range(grid_w)] for _ in range(grid_h)]
    for v in verts:
        col = int((v[0] - x_range[0]) / (x_range[1] - x_range[0]) * (grid_w - 1))
        row = int((1 - (v[1] - y_range[0]) / (y_range[1] - y_range[0])) * (grid_h - 1))
        col = max(0, min(col, grid_w - 1))
        row = max(0, min(row, grid_h - 1))
        canvas[row][col] = '●'

    # Mark center
    cc = int((0 - x_range[0]) / (x_range[1] - x_range[0]) * (grid_w - 1))
    cr = int((1 - (0 - y_range[0]) / (y_range[1] - y_range[0])) * (grid_h - 1))
    if 0 <= cr < grid_h and 0 <= cc < grid_w:
        canvas[cr][cc] = '+'

    print("\n  ASCII plot of zonotope vertices:")
    for row in canvas:
        print("  │" + ''.join(row) + "│")


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run all zonotope demonstrations."""
    print("=" * 60)
    print("  MARACE — Zonotope Abstract Domain Demo")
    print("=" * 60)

    demo_construction()
    demo_operations()
    demo_hb_pruning()
    demo_widening()
    demo_generator_management()
    demo_fixpoint()
    demo_2d_visualization()

    print(f"\n{'=' * 60}")
    print("  Demo complete.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
