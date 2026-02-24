#!/usr/bin/env python3
"""
rust_atomics.py — Rust atomics verification for LITMUS∞.

Verifies Rust std::sync::atomic Ordering usage, detects insufficient ordering,
suggests minimal sufficient ordering, and compares with C++ memory_order equivalents.

Usage:
    from rust_atomics import verify_ordering, suggest_minimal_ordering, compare_with_cpp
    result = verify_ordering("store", "Relaxed", pattern="message_passing", role="sender")
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import IntEnum

sys.path.insert(0, os.path.dirname(__file__))


# ── Ordering Model ──────────────────────────────────────────────────

class RustOrdering(IntEnum):
    """Rust atomic orderings, ranked by strength."""
    Relaxed = 0
    Acquire = 1
    Release = 2
    AcqRel = 3
    SeqCst = 4


# Ordering constraints: which operation pairs each ordering guarantees
ORDERING_GUARANTEES = {
    RustOrdering.Relaxed: set(),
    RustOrdering.Acquire: {('load', 'load'), ('load', 'store')},
    RustOrdering.Release: {('load', 'store'), ('store', 'store')},
    RustOrdering.AcqRel:  {('load', 'load'), ('load', 'store'), ('store', 'store')},
    RustOrdering.SeqCst:  {('load', 'load'), ('load', 'store'), ('store', 'store'), ('store', 'load')},
}

# C++ equivalents
CPP_EQUIVALENTS = {
    RustOrdering.Relaxed: "memory_order_relaxed",
    RustOrdering.Acquire: "memory_order_acquire",
    RustOrdering.Release: "memory_order_release",
    RustOrdering.AcqRel:  "memory_order_acq_rel",
    RustOrdering.SeqCst:  "memory_order_seq_cst",
}

# Valid orderings per operation type in Rust
VALID_ORDERINGS = {
    'store': {RustOrdering.Relaxed, RustOrdering.Release, RustOrdering.SeqCst},
    'load':  {RustOrdering.Relaxed, RustOrdering.Acquire, RustOrdering.SeqCst},
    'rmw':   {RustOrdering.Relaxed, RustOrdering.Acquire, RustOrdering.Release,
              RustOrdering.AcqRel, RustOrdering.SeqCst},
    'fence': {RustOrdering.Acquire, RustOrdering.Release, RustOrdering.AcqRel,
              RustOrdering.SeqCst},
}

# ── Concurrency Patterns ───────────────────────────────────────────

@dataclass
class AtomicOp:
    """An atomic operation in a concurrency pattern."""
    thread: int
    op_type: str       # 'store', 'load', 'rmw', 'fence'
    variable: str
    role: str           # 'sender', 'receiver', 'both'
    min_ordering: RustOrdering
    explanation: str


# Well-known patterns and their minimal ordering requirements
PATTERNS: Dict[str, Dict] = {
    "message_passing": {
        "description": "Producer writes data, sets flag; consumer reads flag, reads data.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "store", "data",  "sender",   RustOrdering.Relaxed,
                     "Data store can be relaxed — flag store provides ordering."),
            AtomicOp(0, "store", "flag",  "sender",   RustOrdering.Release,
                     "Release ensures data store is visible before flag is set."),
            AtomicOp(1, "load",  "flag",  "receiver", RustOrdering.Acquire,
                     "Acquire ensures data load happens after flag is observed."),
            AtomicOp(1, "load",  "data",  "receiver", RustOrdering.Relaxed,
                     "Data load can be relaxed — flag load provides ordering."),
        ],
        "forbidden": "Consumer sees flag=1 but data=0.",
    },
    "spinlock": {
        "description": "Mutex via atomic flag: acquire on lock, release on unlock.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "rmw",   "lock",  "both", RustOrdering.Acquire,
                     "Acquire on lock ensures critical section reads happen after lock."),
            AtomicOp(0, "store", "lock",  "both", RustOrdering.Release,
                     "Release on unlock ensures critical section writes are visible."),
        ],
        "forbidden": "Two threads in critical section simultaneously.",
    },
    "seqlock": {
        "description": "Sequence lock: writer increments seq, writes data, increments seq.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "store", "seq",   "sender",   RustOrdering.Release,
                     "Release on seq update ensures data writes are ordered."),
            AtomicOp(0, "store", "data",  "sender",   RustOrdering.Relaxed,
                     "Data stores between seq updates can be relaxed."),
            AtomicOp(1, "load",  "seq",   "receiver", RustOrdering.Acquire,
                     "Acquire on seq read ensures data reads are ordered."),
            AtomicOp(1, "load",  "data",  "receiver", RustOrdering.Relaxed,
                     "Data loads between seq reads can be relaxed."),
        ],
        "forbidden": "Reader sees inconsistent data (torn read).",
    },
    "reference_counting": {
        "description": "Shared reference count: increment relaxed, decrement release, final acquire.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "rmw",   "refcount", "both", RustOrdering.Relaxed,
                     "Clone (increment) can be relaxed — no data ordering needed."),
            AtomicOp(0, "rmw",   "refcount", "both", RustOrdering.Release,
                     "Drop (decrement) must be release — ensures all prior writes visible."),
            AtomicOp(0, "fence", "refcount", "both", RustOrdering.Acquire,
                     "Acquire fence before deallocation ensures all decrements visible."),
        ],
        "forbidden": "Use-after-free: another thread accesses data after deallocation.",
    },
    "double_checked_locking": {
        "description": "DCL pattern: relaxed check, lock, release store.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "load",  "init",  "receiver", RustOrdering.Acquire,
                     "Acquire on init flag ensures initialization is visible."),
            AtomicOp(0, "store", "init",  "sender",   RustOrdering.Release,
                     "Release on init flag ensures data is fully initialized."),
        ],
        "forbidden": "Thread observes init=true but sees uninitialized data.",
    },
    "spsc_queue": {
        "description": "Single-producer single-consumer bounded queue with head/tail.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "store", "tail", "sender",   RustOrdering.Release,
                     "Release on tail update ensures buffer writes are visible."),
            AtomicOp(0, "load",  "head", "sender",   RustOrdering.Acquire,
                     "Acquire on head read ensures consumer's reads are complete."),
            AtomicOp(1, "store", "head", "receiver", RustOrdering.Release,
                     "Release on head update ensures buffer reads are complete."),
            AtomicOp(1, "load",  "tail", "receiver", RustOrdering.Acquire,
                     "Acquire on tail read ensures producer's writes are visible."),
        ],
        "forbidden": "Consumer reads stale or partially written data.",
    },
    "store_buffer": {
        "description": "Store-buffer pattern (SB): two threads store then load different vars.",
        "threads": 2,
        "ops": [
            AtomicOp(0, "store", "x", "sender",   RustOrdering.SeqCst,
                     "SeqCst needed to prevent store-buffer reordering."),
            AtomicOp(0, "load",  "y", "receiver", RustOrdering.SeqCst,
                     "SeqCst needed — Acquire/Release cannot prevent SB."),
            AtomicOp(1, "store", "y", "sender",   RustOrdering.SeqCst,
                     "SeqCst needed — symmetric with thread 0."),
            AtomicOp(1, "load",  "x", "receiver", RustOrdering.SeqCst,
                     "SeqCst needed — both loads must see at least one store."),
        ],
        "forbidden": "Both threads read 0 (neither sees the other's store).",
    },
}


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class OrderingVerification:
    """Result of verifying a single atomic operation's ordering."""
    op_type: str
    variable: str
    specified_ordering: str
    required_ordering: str
    sufficient: bool
    pattern: str
    role: str
    explanation: str
    overstrength: bool = False  # True if ordering is stronger than needed

    def __repr__(self):
        status = "✓" if self.sufficient else "✗ INSUFFICIENT"
        over = " (overly strong)" if self.overstrength else ""
        return (f"OrderingVerification({self.op_type} {self.variable}: "
                f"{self.specified_ordering} vs required {self.required_ordering} {status}{over})")


@dataclass
class MinimalOrderingSuggestion:
    """Suggestion for minimal sufficient orderings for a pattern."""
    pattern: str
    description: str
    suggestions: List[Dict]  # [{op_type, variable, role, min_ordering, explanation}]
    forbidden_outcome: str

    def __repr__(self):
        return f"MinimalOrdering({self.pattern}: {len(self.suggestions)} ops)"

    def to_text(self) -> str:
        lines = [f"Pattern: {self.pattern}", f"  {self.description}", ""]
        for s in self.suggestions:
            lines.append(f"  {s['op_type']:>5} {s['variable']:<12} → {s['min_ordering']:<8} ({s['role']})")
            lines.append(f"         {s['explanation']}")
        lines.append(f"\n  Prevents: {self.forbidden_outcome}")
        return "\n".join(lines)


@dataclass
class CppComparison:
    """Comparison between Rust Ordering and C++ memory_order."""
    rust_ordering: str
    cpp_equivalent: str
    valid_for: List[str]         # operation types this is valid for
    guarantees: List[str]        # ordering guarantees provided
    differences: List[str]       # subtle differences from C++

    def __repr__(self):
        return f"CppComparison({self.rust_ordering} ↔ {self.cpp_equivalent})"


@dataclass
class PatternCheckResult:
    """Result of checking a full concurrency pattern."""
    pattern: str
    description: str
    verifications: List[OrderingVerification]
    all_sufficient: bool
    has_overstrength: bool
    suggestions: List[str]

    def __repr__(self):
        status = "✓ ALL SUFFICIENT" if self.all_sufficient else "✗ INSUFFICIENT ORDERINGS"
        return f"PatternCheck({self.pattern}: {status})"


# ── Core Logic ──────────────────────────────────────────────────────

def _parse_ordering(name: str) -> RustOrdering:
    """Parse an ordering name string to RustOrdering enum."""
    name_map = {
        'relaxed': RustOrdering.Relaxed,
        'acquire': RustOrdering.Acquire,
        'release': RustOrdering.Release,
        'acqrel': RustOrdering.AcqRel,
        'seqcst': RustOrdering.SeqCst,
    }
    normalized = name.lower().replace('_', '').replace('-', '')
    if normalized not in name_map:
        raise ValueError(f"Unknown ordering '{name}'. Valid: Relaxed, Acquire, Release, AcqRel, SeqCst")
    return name_map[normalized]


def _is_valid_for_op(ordering: RustOrdering, op_type: str) -> bool:
    """Check if an ordering is valid for a given operation type."""
    return ordering in VALID_ORDERINGS.get(op_type, set())


def _find_pattern_op(pattern_name: str, op_type: str, role: str, variable: str = None) -> Optional[AtomicOp]:
    """Find the matching operation in a pattern."""
    if pattern_name not in PATTERNS:
        return None
    for op in PATTERNS[pattern_name]["ops"]:
        if op.op_type == op_type and op.role == role:
            if variable is None or op.variable == variable:
                return op
    # Fallback: match by op_type only
    for op in PATTERNS[pattern_name]["ops"]:
        if op.op_type == op_type:
            return op
    return None


# ── Public API ──────────────────────────────────────────────────────

def verify_ordering(
    op_type: str,
    ordering: str,
    pattern: str = "message_passing",
    role: str = "sender",
    variable: str = None,
) -> OrderingVerification:
    """
    Verify whether a Rust atomic ordering is sufficient for a given role in a pattern.

    Args:
        op_type: Operation type ("store", "load", "rmw", "fence").
        ordering: Rust ordering name ("Relaxed", "Acquire", "Release", "AcqRel", "SeqCst").
        pattern: Concurrency pattern name.
        role: Role in the pattern ("sender", "receiver", "both").
        variable: Optional variable name to match specific op in the pattern.

    Returns:
        OrderingVerification with sufficiency analysis.
    """
    specified = _parse_ordering(ordering)

    if not _is_valid_for_op(specified, op_type):
        return OrderingVerification(
            op_type=op_type,
            variable=variable or "?",
            specified_ordering=ordering,
            required_ordering="N/A",
            sufficient=False,
            pattern=pattern,
            role=role,
            explanation=f"Ordering '{ordering}' is not valid for {op_type} operations in Rust. "
                        f"Valid orderings: {', '.join(o.name for o in VALID_ORDERINGS[op_type])}.",
        )

    if pattern not in PATTERNS:
        available = ', '.join(sorted(PATTERNS.keys()))
        return OrderingVerification(
            op_type=op_type,
            variable=variable or "?",
            specified_ordering=ordering,
            required_ordering="unknown",
            sufficient=True,  # Can't determine without pattern
            pattern=pattern,
            role=role,
            explanation=f"Unknown pattern '{pattern}'. Available: {available}. "
                        f"Cannot verify sufficiency without pattern definition.",
        )

    match = _find_pattern_op(pattern, op_type, role, variable)
    if match is None:
        return OrderingVerification(
            op_type=op_type,
            variable=variable or "?",
            specified_ordering=ordering,
            required_ordering="N/A",
            sufficient=True,
            pattern=pattern,
            role=role,
            explanation=f"No matching {op_type}/{role} operation found in pattern '{pattern}'.",
        )

    required = match.min_ordering
    sufficient = specified >= required
    overstrength = specified > required

    if sufficient:
        explanation = match.explanation
        if overstrength:
            explanation += f" ('{ordering}' is stronger than needed; '{required.name}' suffices.)"
    else:
        explanation = (
            f"'{ordering}' is insufficient for {op_type} in {pattern}/{role}. "
            f"Minimum required: '{required.name}'. {match.explanation}"
        )

    return OrderingVerification(
        op_type=op_type,
        variable=match.variable,
        specified_ordering=ordering,
        required_ordering=required.name,
        sufficient=sufficient,
        pattern=pattern,
        role=role,
        explanation=explanation,
        overstrength=overstrength,
    )


def suggest_minimal_ordering(pattern: str) -> MinimalOrderingSuggestion:
    """
    Suggest minimal sufficient orderings for all operations in a pattern.

    Args:
        pattern: Concurrency pattern name.

    Returns:
        MinimalOrderingSuggestion with per-operation recommendations.
    """
    if pattern not in PATTERNS:
        available = ', '.join(sorted(PATTERNS.keys()))
        raise ValueError(f"Unknown pattern '{pattern}'. Available: {available}")

    pat = PATTERNS[pattern]
    suggestions = []

    for op in pat["ops"]:
        suggestions.append({
            "thread": op.thread,
            "op_type": op.op_type,
            "variable": op.variable,
            "role": op.role,
            "min_ordering": op.min_ordering.name,
            "explanation": op.explanation,
        })

    return MinimalOrderingSuggestion(
        pattern=pattern,
        description=pat["description"],
        suggestions=suggestions,
        forbidden_outcome=pat["forbidden"],
    )


def compare_with_cpp(ordering: str) -> CppComparison:
    """
    Compare a Rust atomic ordering with its C++ memory_order equivalent.

    Args:
        ordering: Rust ordering name.

    Returns:
        CppComparison with equivalence and differences.
    """
    rust_ord = _parse_ordering(ordering)
    cpp_eq = CPP_EQUIVALENTS[rust_ord]

    valid_for = [op for op, ords in VALID_ORDERINGS.items() if rust_ord in ords]
    guarantees = []
    for pair in sorted(ORDERING_GUARANTEES[rust_ord]):
        guarantees.append(f"{pair[0]}→{pair[1]}")

    differences = []
    if rust_ord == RustOrdering.Relaxed:
        differences.append(
            "Identical semantics. Both provide only atomicity, no ordering guarantees."
        )
    elif rust_ord == RustOrdering.Acquire:
        differences.append(
            "Identical semantics. In Rust, Acquire is only valid for loads (not stores); "
            "C++ allows memory_order_acquire on stores but it's meaningless."
        )
        differences.append(
            "Rust enforces at compile time that Acquire cannot be used with store(), "
            "while C++ only warns."
        )
    elif rust_ord == RustOrdering.Release:
        differences.append(
            "Identical semantics. In Rust, Release is only valid for stores (not loads); "
            "C++ allows memory_order_release on loads but it's meaningless."
        )
        differences.append(
            "Rust enforces at compile time that Release cannot be used with load(), "
            "while C++ only warns."
        )
    elif rust_ord == RustOrdering.AcqRel:
        differences.append(
            "Identical semantics. In Rust, AcqRel is only valid for RMW operations; "
            "C++ allows it on plain loads/stores where it degrades to Acquire or Release."
        )
    elif rust_ord == RustOrdering.SeqCst:
        differences.append(
            "Identical semantics. Both establish a single total order for all SeqCst operations."
        )
        differences.append(
            "Performance note: SeqCst typically requires a full memory barrier (MFENCE on x86, "
            "dmb ish on ARM, fence rw,rw on RISC-V)."
        )

    return CppComparison(
        rust_ordering=rust_ord.name,
        cpp_equivalent=cpp_eq,
        valid_for=valid_for,
        guarantees=guarantees if guarantees else ["(atomicity only, no ordering)"],
        differences=differences,
    )


def check_pattern(
    pattern: str,
    orderings: Dict[str, str] = None,
) -> PatternCheckResult:
    """
    Check a full pattern with user-specified orderings.

    Args:
        pattern: Concurrency pattern name.
        orderings: Dict mapping "op_type:variable" or "op_type:role" to ordering name.
                   If None, checks with minimal required orderings.

    Returns:
        PatternCheckResult with per-operation verification.
    """
    if pattern not in PATTERNS:
        raise ValueError(f"Unknown pattern '{pattern}'.")

    pat = PATTERNS[pattern]
    verifications = []
    all_sufficient = True
    has_overstrength = False
    suggestions = []

    for op in pat["ops"]:
        key_var = f"{op.op_type}:{op.variable}"
        key_role = f"{op.op_type}:{op.role}"

        if orderings:
            user_ordering = orderings.get(key_var) or orderings.get(key_role)
            if user_ordering is None:
                # Default to the minimal required
                user_ordering = op.min_ordering.name
        else:
            user_ordering = op.min_ordering.name

        v = verify_ordering(op.op_type, user_ordering, pattern, op.role, op.variable)
        verifications.append(v)

        if not v.sufficient:
            all_sufficient = False
            suggestions.append(
                f"Change {op.op_type}({op.variable}) from {user_ordering} to {op.min_ordering.name}"
            )
        if v.overstrength:
            has_overstrength = True
            suggestions.append(
                f"Can weaken {op.op_type}({op.variable}) from {user_ordering} to {op.min_ordering.name}"
            )

    return PatternCheckResult(
        pattern=pattern,
        description=pat["description"],
        verifications=verifications,
        all_sufficient=all_sufficient,
        has_overstrength=has_overstrength,
        suggestions=suggestions,
    )


def list_patterns() -> List[Dict]:
    """List all known concurrency patterns with their descriptions."""
    return [
        {
            "name": name,
            "description": pat["description"],
            "threads": pat["threads"],
            "ops_count": len(pat["ops"]),
            "forbidden": pat["forbidden"],
        }
        for name, pat in sorted(PATTERNS.items())
    ]


def ordering_strength_table() -> str:
    """Return a formatted table showing ordering strengths and valid operations."""
    lines = [
        "Rust Ordering Strength Table",
        "=" * 70,
        f"{'Ordering':<12} {'Strength':>8}  {'Valid For':<20} {'Guarantees':<25}",
        "-" * 70,
    ]
    for ordering in RustOrdering:
        valid = ', '.join(op for op, ords in VALID_ORDERINGS.items() if ordering in ords)
        guarantees = ORDERING_GUARANTEES[ordering]
        g_str = ', '.join(f"{a}→{b}" for a, b in sorted(guarantees)) if guarantees else "(none)"
        lines.append(f"{ordering.name:<12} {ordering.value:>8}  {valid:<20} {g_str:<25}")
    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="LITMUS∞ Rust Atomics Verifier")
    sub = parser.add_subparsers(dest="cmd")

    p_verify = sub.add_parser("verify", help="Verify ordering sufficiency")
    p_verify.add_argument("op_type", choices=["store", "load", "rmw", "fence"])
    p_verify.add_argument("ordering")
    p_verify.add_argument("--pattern", default="message_passing")
    p_verify.add_argument("--role", default="sender")
    p_verify.add_argument("--variable")

    p_suggest = sub.add_parser("suggest", help="Suggest minimal orderings")
    p_suggest.add_argument("pattern")

    p_compare = sub.add_parser("compare", help="Compare with C++")
    p_compare.add_argument("ordering")

    p_check = sub.add_parser("check", help="Check pattern with orderings")
    p_check.add_argument("pattern")
    p_check.add_argument("--orderings", nargs="*", help="op:var=Ordering pairs")

    p_list = sub.add_parser("list", help="List patterns")
    p_table = sub.add_parser("table", help="Show ordering strength table")

    args = parser.parse_args()

    if args.cmd == "verify":
        result = verify_ordering(args.op_type, args.ordering, args.pattern, args.role, args.variable)
        print(result)
        print(f"  {result.explanation}")
    elif args.cmd == "suggest":
        suggestion = suggest_minimal_ordering(args.pattern)
        print(suggestion.to_text())
    elif args.cmd == "compare":
        cmp = compare_with_cpp(args.ordering)
        print(cmp)
        print(f"  C++: {cmp.cpp_equivalent}")
        print(f"  Valid for: {', '.join(cmp.valid_for)}")
        print(f"  Guarantees: {', '.join(cmp.guarantees)}")
        for d in cmp.differences:
            print(f"  • {d}")
    elif args.cmd == "check":
        ords = {}
        if args.orderings:
            for pair in args.orderings:
                k, v = pair.split("=")
                ords[k] = v
        result = check_pattern(args.pattern, ords if ords else None)
        print(result)
        for v in result.verifications:
            print(f"  {v}")
        if result.suggestions:
            print("\nSuggestions:")
            for s in result.suggestions:
                print(f"  → {s}")
    elif args.cmd == "list":
        for p in list_patterns():
            print(f"  {p['name']:<30} {p['description']}")
    elif args.cmd == "table":
        print(ordering_strength_table())
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
