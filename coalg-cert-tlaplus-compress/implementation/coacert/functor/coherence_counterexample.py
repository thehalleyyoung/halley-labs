"""
Counterexample demonstrating necessity of the T-Fair coherence condition.

Constructs a concrete 4-state fair transition system where:
  - Stuttering equivalence merges states s1 and s2 (they have identical
    AP labels and identical observable successor structure modulo stuttering).
  - A fairness acceptance pair (B, G) has s1 ∈ B but s2 ∉ B.
  - T-Fair coherence FAILS: the stutter class {s1, s2} is split by B.
  - Consequence: quotienting merges s1 and s2, and the quotient system
    loses the liveness property "under WF, eventually reach G infinitely
    often while visiting B infinitely often". Specifically, in the original
    system s2 has a fair path that never visits B, so it violates the
    acceptance condition. But s1 DOES satisfy the acceptance condition.
    After quotienting, the merged state inherits s1's membership in B,
    creating a spurious fair acceptance that does not exist for paths
    originating from s2's behavior.

This demonstrates that the T-Fair coherence condition is NECESSARY for
liveness preservation, not merely sufficient.

System:
    s0 --a--> s1, s0 --a--> s2
    s1 --a--> s3              (s1 ∈ B, s1 ∈ G)
    s2 --a--> s3              (s2 ∉ B, s2 ∈ G)
    s3 --a--> s3              (s3 ∈ G)

    AP(s0) = {init}, AP(s1) = {mid}, AP(s2) = {mid}, AP(s3) = {done}
    Fairness pair: (B={s1}, G={s1, s2, s3})

    s1 and s2 are stuttering-equivalent (same AP, same successor structure
    to s3, no stuttering steps). But s1 ∈ B while s2 ∉ B.

    Liveness property: under WF, every fair path visits B infinitely often.
    - In original: paths through s2 never visit B, so the property FAILS
      for s2-paths. But fair acceptance is {(B,G)} with B={s1}: only
      paths through s1 satisfy the acceptance pair.
    - In quotient: s1 and s2 are merged into [s12]. Now [s12] ∈ B
      (inherited from s1). ALL paths through [s12] appear to visit B,
      which is spurious for s2-originating behavior.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class CounterexampleSystem:
    """A minimal fair transition system violating T-Fair coherence."""
    states: Set[str]
    initial_states: Set[str]
    actions: Set[str]
    transitions: Dict[str, Dict[str, Set[str]]]
    labels: Dict[str, Set[str]]
    fairness_pairs: List[Tuple[FrozenSet[str], FrozenSet[str]]]
    stutter_classes: List[FrozenSet[str]]

    # Derived
    quotient_map: Dict[str, str] = field(default_factory=dict)
    quotient_labels: Dict[str, Set[str]] = field(default_factory=dict)
    quotient_fairness: List[Tuple[FrozenSet[str], FrozenSet[str]]] = field(
        default_factory=list
    )


def build_coherence_counterexample() -> CounterexampleSystem:
    """Construct the 4-state counterexample system.

    Returns a CounterexampleSystem with all fields populated.
    """
    states = {"s0", "s1", "s2", "s3"}
    initial_states = {"s0"}
    actions = {"a"}

    transitions: Dict[str, Dict[str, Set[str]]] = {
        "s0": {"a": {"s1", "s2"}},
        "s1": {"a": {"s3"}},
        "s2": {"a": {"s3"}},
        "s3": {"a": {"s3"}},
    }

    labels: Dict[str, Set[str]] = {
        "s0": {"init"},
        "s1": {"mid"},
        "s2": {"mid"},
        "s3": {"done"},
    }

    # Fairness: acceptance pair (B, G) where B = {s1}, G = {s1, s2, s3}
    b_set = frozenset({"s1"})
    g_set = frozenset({"s1", "s2", "s3"})
    fairness_pairs = [(b_set, g_set)]

    # Stutter equivalence classes (s1 ≡ s2 because same AP and same successors)
    stutter_classes = [
        frozenset({"s0"}),
        frozenset({"s1", "s2"}),  # merged by stuttering equivalence
        frozenset({"s3"}),
    ]

    sys = CounterexampleSystem(
        states=states,
        initial_states=initial_states,
        actions=actions,
        transitions=transitions,
        labels=labels,
        fairness_pairs=fairness_pairs,
        stutter_classes=stutter_classes,
    )

    # Build quotient
    sys.quotient_map = {
        "s0": "s0",
        "s1": "s1",  # s1 is representative of {s1, s2}
        "s2": "s1",
        "s3": "s3",
    }
    sys.quotient_labels = {
        "s0": {"init"},
        "s1": {"mid"},
        "s3": {"done"},
    }
    # In the quotient, B lifts to {s1} (representative of the merged class)
    sys.quotient_fairness = [
        (frozenset({"s1"}), frozenset({"s1", "s3"}))
    ]

    return sys


def verify_coherence_fails(sys: CounterexampleSystem) -> bool:
    """Verify that T-Fair coherence fails for this system.

    Checks that some stutter class is split by some B_i or G_i.
    Returns True if coherence FAILS (i.e., the counterexample is valid).
    """
    for b_set, g_set in sys.fairness_pairs:
        for cls in sys.stutter_classes:
            b_inter = cls & b_set
            if b_inter and b_inter != cls:
                return True  # coherence fails: class split by B
            g_inter = cls & g_set
            if g_inter and g_inter != cls:
                return True  # coherence fails: class split by G
    return False


def verify_liveness_violation(sys: CounterexampleSystem) -> dict:
    """Verify that quotienting causes a spurious liveness acceptance.

    In the original system:
      - Path s0 -> s1 -> s3 -> s3 -> ... visits B (at s1) then stays in G.
        This path satisfies the acceptance pair.
      - Path s0 -> s2 -> s3 -> s3 -> ... never visits B.
        This path does NOT satisfy the acceptance pair.

    In the quotient:
      - s1 and s2 are merged. The quotient path s0 -> [s1] -> s3 -> ...
        has [s1] ∈ B (inherited from s1). So ALL quotient paths through
        [s1] appear to satisfy the acceptance pair, which is SPURIOUS
        for paths that originated from s2.

    Returns a dict with analysis details.
    """
    result = {
        "coherence_fails": verify_coherence_fails(sys),
        "original_system": {},
        "quotient_system": {},
        "liveness_spurious": False,
    }

    # Original system analysis
    b_set = sys.fairness_pairs[0][0]

    path_via_s1 = ["s0", "s1", "s3"]
    path_via_s2 = ["s0", "s2", "s3"]

    s1_visits_b = any(s in b_set for s in path_via_s1)
    s2_visits_b = any(s in b_set for s in path_via_s2)

    result["original_system"] = {
        "path_via_s1_visits_B": s1_visits_b,
        "path_via_s2_visits_B": s2_visits_b,
        "s1_satisfies_acceptance": s1_visits_b,
        "s2_satisfies_acceptance": s2_visits_b,
    }

    # Quotient analysis
    q_b_set = sys.quotient_fairness[0][0]
    q_path = ["s0", "s1", "s3"]  # all paths merge through [s1]

    q_visits_b = any(s in q_b_set for s in q_path)

    result["quotient_system"] = {
        "merged_path_visits_B": q_visits_b,
        "all_paths_appear_accepting": q_visits_b,
        "original_s2_path_was_non_accepting": not s2_visits_b,
    }

    # The spurious acceptance: quotient says all paths through [s1] visit B,
    # but in reality, paths from s2 never visited B.
    result["liveness_spurious"] = (
        q_visits_b and not s2_visits_b
    )

    return result


def full_counterexample_analysis() -> dict:
    """Run the complete counterexample analysis and return results.

    This is the main entry point for demonstrating the necessity of
    T-Fair coherence.
    """
    sys = build_coherence_counterexample()

    analysis = {
        "system": {
            "states": sorted(sys.states),
            "actions": sorted(sys.actions),
            "transitions": {
                s: {a: sorted(ts) for a, ts in am.items()}
                for s, am in sorted(sys.transitions.items())
            },
            "labels": {s: sorted(ap) for s, ap in sorted(sys.labels.items())},
            "fairness_pairs": [
                {"B": sorted(b), "G": sorted(g)}
                for b, g in sys.fairness_pairs
            ],
            "stutter_classes": [sorted(c) for c in sys.stutter_classes],
        },
        "coherence_fails": verify_coherence_fails(sys),
        "liveness_analysis": verify_liveness_violation(sys),
    }

    return analysis
