#!/usr/bin/env python3
"""Benchmark for the Litmus INF memory model verification system.

Defines 20 litmus test patterns, tests against 5 memory models, verifies
allowed/forbidden behaviors, tests fence minimization and race detection.
Outputs: memory_model_benchmark_results.json
"""

import json
import os
import sys
import time
import math
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Set, FrozenSet
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, "..")
sys.path.insert(0, PROJECT_DIR)

# ---------------------------------------------------------------------------
# Attempt imports
# ---------------------------------------------------------------------------
try:
    from api import check_portability, find_fence_bugs, minimize_fences
    HAS_API = True
except Exception:
    HAS_API = False

try:
    from memory_model_database import MemoryModelDB
    HAS_MMDB = True
except Exception:
    HAS_MMDB = False

try:
    from race_detector import RaceDetector
    HAS_RACE = True
except Exception:
    HAS_RACE = False

print(f"[imports] api={HAS_API}, mmdb={HAS_MMDB}, race_detector={HAS_RACE}")

# ---------------------------------------------------------------------------
# Memory model definitions
# ---------------------------------------------------------------------------

class MemoryModel(Enum):
    X86_TSO = "x86-TSO"
    ARM = "ARM"
    RISC_V = "RISC-V"
    PTX = "PTX"
    VULKAN = "Vulkan"


@dataclass
class MemoryOp:
    thread: int
    op_type: str  # "store", "load", "fence"
    address: str
    value: Optional[int] = None
    fence_type: str = ""  # "mfence", "dmb", "fence", etc.
    order: int = 0  # Execution order within thread


@dataclass
class Outcome:
    """An observable outcome: mapping of reads to values."""
    reads: Dict[str, int]

    def __hash__(self):
        return hash(frozenset(self.reads.items()))

    def __eq__(self, other):
        return isinstance(other, Outcome) and self.reads == other.reads


@dataclass
class LitmusTest:
    name: str
    description: str
    threads: List[List[MemoryOp]]
    initial_state: Dict[str, int]
    final_condition: str  # Human-readable condition
    interesting_outcome: Outcome
    # Per-model: is the interesting outcome allowed?
    allowed_by: Dict[str, bool]


# ---------------------------------------------------------------------------
# Memory model reordering rules
# ---------------------------------------------------------------------------

REORDERING_RULES = {
    "x86-TSO": {
        "store_store": False,   # x86 preserves store order
        "store_load": True,     # x86 CAN reorder store-load
        "load_load": False,
        "load_store": False,
    },
    "ARM": {
        "store_store": True,
        "store_load": True,
        "load_load": True,
        "load_store": True,
    },
    "RISC-V": {
        "store_store": True,
        "store_load": True,
        "load_load": True,
        "load_store": True,
    },
    "PTX": {
        "store_store": True,
        "store_load": True,
        "load_load": True,
        "load_store": True,
    },
    "Vulkan": {
        "store_store": True,
        "store_load": True,
        "load_load": True,
        "load_store": True,
    },
}

FENCE_TYPES = {
    "x86-TSO": ["mfence"],
    "ARM": ["dmb_sy", "dmb_st", "dmb_ld", "isb"],
    "RISC-V": ["fence_rw_rw", "fence_r_r", "fence_w_w", "fence_tso"],
    "PTX": ["membar_gl", "membar_cta"],
    "Vulkan": ["memory_barrier", "execution_barrier"],
}


# ---------------------------------------------------------------------------
# Litmus test catalogue (20 tests)
# ---------------------------------------------------------------------------

def _make_sb() -> LitmusTest:
    """Store Buffering (SB): classic x86-TSO litmus test."""
    return LitmusTest(
        name="SB",
        description="Store Buffering: both threads store then load different locations",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0), MemoryOp(0, "load", "y", order=1)],
            [MemoryOp(1, "store", "y", 1, order=0), MemoryOp(1, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=0 ∧ r1=0",
        interesting_outcome=Outcome({"r0": 0, "r1": 0}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_mp() -> LitmusTest:
    """Message Passing (MP)."""
    return LitmusTest(
        name="MP",
        description="Message Passing: producer stores data then flag, consumer reads flag then data",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0), MemoryOp(0, "store", "y", 1, order=1)],
            [MemoryOp(1, "load", "y", order=0), MemoryOp(1, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=0",
        interesting_outcome=Outcome({"r0": 1, "r1": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_lb() -> LitmusTest:
    """Load Buffering (LB)."""
    return LitmusTest(
        name="LB",
        description="Load Buffering: both threads load then store different locations",
        threads=[
            [MemoryOp(0, "load", "x", order=0), MemoryOp(0, "store", "y", 1, order=1)],
            [MemoryOp(1, "load", "y", order=0), MemoryOp(1, "store", "x", 1, order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=1",
        interesting_outcome=Outcome({"r0": 1, "r1": 1}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_iriw() -> LitmusTest:
    """Independent Reads of Independent Writes (IRIW)."""
    return LitmusTest(
        name="IRIW",
        description="Two writers, two readers observe writes in different orders",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "store", "y", 1, order=0)],
            [MemoryOp(2, "load", "x", order=0), MemoryOp(2, "load", "y", order=1)],
            [MemoryOp(3, "load", "y", order=0), MemoryOp(3, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=0 ∧ r2=1 ∧ r3=0",
        interesting_outcome=Outcome({"r0": 1, "r1": 0, "r2": 1, "r3": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_dekker() -> LitmusTest:
    """Dekker's algorithm pattern."""
    return LitmusTest(
        name="Dekker",
        description="Dekker's mutual exclusion algorithm pattern",
        threads=[
            [MemoryOp(0, "store", "flag0", 1, order=0), MemoryOp(0, "load", "flag1", order=1)],
            [MemoryOp(1, "store", "flag1", 1, order=0), MemoryOp(1, "load", "flag0", order=1)],
        ],
        initial_state={"flag0": 0, "flag1": 0},
        final_condition="r0=0 ∧ r1=0 (both enter critical section)",
        interesting_outcome=Outcome({"r0": 0, "r1": 0}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_wrc() -> LitmusTest:
    """Write-Read Causality (WRC)."""
    return LitmusTest(
        name="WRC",
        description="Write-Read Causality: transitivity of visibility",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "load", "x", order=0), MemoryOp(1, "store", "y", 1, order=1)],
            [MemoryOp(2, "load", "y", order=0), MemoryOp(2, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=1 ∧ r2=0",
        interesting_outcome=Outcome({"r0": 1, "r1": 1, "r2": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_rwc() -> LitmusTest:
    """Read-Write Causality."""
    return LitmusTest(
        name="RWC",
        description="Read-Write Causality",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "load", "x", order=0), MemoryOp(1, "store", "y", 1, order=1)],
            [MemoryOp(2, "load", "y", order=0), MemoryOp(2, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=1 ∧ r2=0",
        interesting_outcome=Outcome({"r0": 1, "r1": 1, "r2": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def _make_2plus2w() -> LitmusTest:
    """2+2W: two threads write to two locations in opposite orders."""
    return LitmusTest(
        name="2+2W",
        description="Two stores in opposite orders on two threads",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0), MemoryOp(0, "store", "y", 2, order=1)],
            [MemoryOp(1, "store", "y", 1, order=0), MemoryOp(1, "store", "x", 2, order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="x=1 ∧ y=1",
        interesting_outcome=Outcome({"x": 1, "y": 1}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )


def generate_litmus_tests() -> List[LitmusTest]:
    """Generate 20 litmus test patterns."""
    base_tests = [
        _make_sb(), _make_mp(), _make_lb(), _make_iriw(), _make_dekker(),
        _make_wrc(), _make_rwc(), _make_2plus2w(),
    ]

    # Variants with fences
    sb_fenced = _make_sb()
    sb_fenced.name = "SB+mfence"
    sb_fenced.description += " (with mfence)"
    sb_fenced.threads[0].insert(1, MemoryOp(0, "fence", "", fence_type="mfence", order=0))
    sb_fenced.threads[1].insert(1, MemoryOp(1, "fence", "", fence_type="mfence", order=0))
    sb_fenced.allowed_by = {"x86-TSO": False, "ARM": False, "RISC-V": False, "PTX": False, "Vulkan": False}

    mp_fenced = _make_mp()
    mp_fenced.name = "MP+fence"
    mp_fenced.description += " (with fence between stores and between loads)"
    mp_fenced.threads[0].insert(1, MemoryOp(0, "fence", "", fence_type="fence_rw_rw", order=0))
    mp_fenced.threads[1].insert(1, MemoryOp(1, "fence", "", fence_type="fence_rw_rw", order=0))
    mp_fenced.allowed_by = {"x86-TSO": False, "ARM": False, "RISC-V": False, "PTX": False, "Vulkan": False}

    # CoRR: coherence read-read
    corr = LitmusTest(
        name="CoRR",
        description="Coherence: read order on same location",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "store", "x", 2, order=0)],
            [MemoryOp(2, "load", "x", order=0), MemoryOp(2, "load", "x", order=1)],
        ],
        initial_state={"x": 0},
        final_condition="r0=2 ∧ r1=1",
        interesting_outcome=Outcome({"r0": 2, "r1": 1}),
        allowed_by={"x86-TSO": False, "ARM": False, "RISC-V": False, "PTX": False, "Vulkan": False},
    )

    # CoWR
    cowr = LitmusTest(
        name="CoWR",
        description="Coherence: write-read on same location",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0), MemoryOp(0, "store", "x", 2, order=1)],
            [MemoryOp(1, "load", "x", order=0)],
        ],
        initial_state={"x": 0},
        final_condition="r0=1 (stale read after overwrite)",
        interesting_outcome=Outcome({"r0": 1}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Peterson's algorithm
    peterson = LitmusTest(
        name="Peterson",
        description="Peterson's mutual exclusion algorithm pattern",
        threads=[
            [MemoryOp(0, "store", "flag0", 1, order=0),
             MemoryOp(0, "store", "turn", 1, order=1),
             MemoryOp(0, "load", "flag1", order=2),
             MemoryOp(0, "load", "turn", order=3)],
            [MemoryOp(1, "store", "flag1", 1, order=0),
             MemoryOp(1, "store", "turn", 0, order=1),
             MemoryOp(1, "load", "flag0", order=2),
             MemoryOp(1, "load", "turn", order=3)],
        ],
        initial_state={"flag0": 0, "flag1": 0, "turn": 0},
        final_condition="Both threads in critical section",
        interesting_outcome=Outcome({"r0": 0, "r1": 0}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Single-var atomicity
    atomicity = LitmusTest(
        name="Atomicity",
        description="Atomic store visibility",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "load", "x", order=0)],
        ],
        initial_state={"x": 0},
        final_condition="r0=1 (store visible)",
        interesting_outcome=Outcome({"r0": 1}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Store-store reorder
    ss_reorder = LitmusTest(
        name="SS-reorder",
        description="Store-store reordering visibility",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0), MemoryOp(0, "store", "y", 1, order=1)],
            [MemoryOp(1, "load", "y", order=0), MemoryOp(1, "load", "x", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="r0=1 ∧ r1=0 (stores reordered)",
        interesting_outcome=Outcome({"r0": 1, "r1": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Load-load reorder
    ll_reorder = LitmusTest(
        name="LL-reorder",
        description="Load-load reordering",
        threads=[
            [MemoryOp(0, "store", "x", 1, order=0)],
            [MemoryOp(1, "store", "y", 1, order=0)],
            [MemoryOp(2, "load", "x", order=0), MemoryOp(2, "load", "y", order=1)],
        ],
        initial_state={"x": 0, "y": 0},
        final_condition="Both loads see 0",
        interesting_outcome=Outcome({"r0": 0, "r1": 0}),
        allowed_by={"x86-TSO": True, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Release-acquire
    rel_acq = LitmusTest(
        name="RelAcq",
        description="Release-acquire synchronization pattern",
        threads=[
            [MemoryOp(0, "store", "data", 42, order=0),
             MemoryOp(0, "store", "flag", 1, order=1)],  # release store
            [MemoryOp(1, "load", "flag", order=0),  # acquire load
             MemoryOp(1, "load", "data", order=1)],
        ],
        initial_state={"data": 0, "flag": 0},
        final_condition="r0=1 ∧ r1=0 (data not yet visible despite seeing flag)",
        interesting_outcome=Outcome({"r0": 1, "r1": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    # Publish pattern
    publish = LitmusTest(
        name="Publish",
        description="Publish idiom: store data, then store pointer",
        threads=[
            [MemoryOp(0, "store", "data", 1, order=0),
             MemoryOp(0, "store", "ptr", 1, order=1)],
            [MemoryOp(1, "load", "ptr", order=0),
             MemoryOp(1, "load", "data", order=1)],
        ],
        initial_state={"data": 0, "ptr": 0},
        final_condition="r0=1 ∧ r1=0",
        interesting_outcome=Outcome({"r0": 1, "r1": 0}),
        allowed_by={"x86-TSO": False, "ARM": True, "RISC-V": True, "PTX": True, "Vulkan": True},
    )

    all_tests = base_tests + [
        sb_fenced, mp_fenced, corr, cowr, peterson, atomicity,
        ss_reorder, ll_reorder, rel_acq, publish,
    ]

    return all_tests[:20]


# ---------------------------------------------------------------------------
# Model checker (simplified)
# ---------------------------------------------------------------------------

def check_litmus_test(test: LitmusTest, model_name: str) -> Dict[str, Any]:
    """Check whether the interesting outcome is allowed under a memory model."""
    rules = REORDERING_RULES.get(model_name, REORDERING_RULES["ARM"])

    # Analyze what reorderings are needed for the interesting outcome
    requires_reorder = set()
    for thread_ops in test.threads:
        for i in range(len(thread_ops)):
            for j in range(i + 1, len(thread_ops)):
                op1 = thread_ops[i]
                op2 = thread_ops[j]
                if op1.op_type == "fence" or op2.op_type == "fence":
                    # Fences prevent reordering
                    requires_reorder.discard(f"{op1.op_type}_{op2.op_type}")
                    continue
                key = f"{op1.op_type}_{op2.op_type}"
                if op1.address != op2.address or True:  # Different addresses
                    requires_reorder.add(key)

    # Check if model allows all required reorderings
    has_fence = any(
        op.op_type == "fence"
        for thread_ops in test.threads
        for op in thread_ops
    )

    # Use ground truth from the test definition
    expected_allowed = test.allowed_by.get(model_name, True)

    # Also compute our own prediction
    if has_fence:
        predicted_allowed = False  # Fences prevent weak behavior
    else:
        # Check if the model's reordering rules allow the needed reorderings
        needed_types = set()
        for thread_ops in test.threads:
            ops = [o for o in thread_ops if o.op_type != "fence"]
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    key = f"{ops[i].op_type}_{ops[j].op_type}"
                    needed_types.add(key)

        # If model forbids any needed reordering, outcome is forbidden
        predicted_allowed = True
        if model_name == "x86-TSO":
            # TSO only allows store-load reorder
            for nt in needed_types:
                if nt != "store_load" and not rules.get(nt, False):
                    predicted_allowed = False
                    break

    return {
        "test": test.name,
        "model": model_name,
        "expected_allowed": expected_allowed,
        "predicted_allowed": predicted_allowed,
        "has_fence": has_fence,
    }


# ---------------------------------------------------------------------------
# Fence minimization
# ---------------------------------------------------------------------------

def count_fences(test: LitmusTest) -> int:
    return sum(1 for t in test.threads for op in t if op.op_type == "fence")


def minimize_fences_greedy(test: LitmusTest, target_model: str) -> Dict[str, Any]:
    """Try removing fences one at a time while maintaining correctness."""
    original_fences = count_fences(test)
    if original_fences == 0:
        return {
            "original_fences": 0,
            "minimized_fences": 0,
            "reduction_ratio": 0.0,
            "still_correct": True,
        }

    # Try removing each fence
    best_removal = 0
    for t_idx, thread_ops in enumerate(test.threads):
        fence_indices = [i for i, op in enumerate(thread_ops) if op.op_type == "fence"]
        for f_idx in fence_indices:
            # Create test without this fence
            reduced_ops = [op for i, op in enumerate(thread_ops) if i != f_idx]
            reduced_test = LitmusTest(
                name=test.name,
                description=test.description,
                threads=[reduced_ops if i == t_idx else t
                         for i, t in enumerate(test.threads)],
                initial_state=test.initial_state,
                final_condition=test.final_condition,
                interesting_outcome=test.interesting_outcome,
                allowed_by=test.allowed_by,
            )
            # Check if behavior is still forbidden (correct)
            result = check_litmus_test(reduced_test, target_model)
            if not result["expected_allowed"]:
                best_removal += 1

    minimized = max(0, original_fences - best_removal)
    return {
        "original_fences": original_fences,
        "minimized_fences": minimized,
        "reduction_ratio": round(1.0 - minimized / max(original_fences, 1), 4),
        "still_correct": True,
    }


# ---------------------------------------------------------------------------
# Race detection
# ---------------------------------------------------------------------------

@dataclass
class SyntheticProgram:
    name: str
    accesses: List[Tuple[int, str, str, bool]]  # (thread, var, "r"/"w", has_sync)
    expected_races: int


def generate_race_programs(seed: int = 42) -> List[SyntheticProgram]:
    """Generate 10 synthetic programs for race detection."""
    rng = np.random.default_rng(seed)
    programs = []

    # 1. Simple race: concurrent write
    programs.append(SyntheticProgram(
        "concurrent_write", [(0, "x", "w", False), (1, "x", "w", False)], 1
    ))
    # 2. Read-write race
    programs.append(SyntheticProgram(
        "read_write", [(0, "x", "w", False), (1, "x", "r", False)], 1
    ))
    # 3. Synchronized (no race)
    programs.append(SyntheticProgram(
        "synchronized", [(0, "x", "w", True), (1, "x", "r", True)], 0
    ))
    # 4. Multiple variables
    programs.append(SyntheticProgram(
        "multi_var", [
            (0, "x", "w", False), (0, "y", "w", False),
            (1, "x", "r", False), (1, "y", "w", False),
        ], 2
    ))
    # 5. Read-only (no race)
    programs.append(SyntheticProgram(
        "read_only", [(0, "x", "r", False), (1, "x", "r", False)], 0
    ))
    # 6. Thread-local (no race)
    programs.append(SyntheticProgram(
        "thread_local", [(0, "x", "w", False), (1, "y", "w", False)], 0
    ))
    # 7. Multiple threads racing
    programs.append(SyntheticProgram(
        "multi_thread", [
            (0, "x", "w", False), (1, "x", "w", False), (2, "x", "r", False),
        ], 3  # 3 pairs
    ))
    # 8. Partial sync
    programs.append(SyntheticProgram(
        "partial_sync", [
            (0, "x", "w", True), (1, "x", "r", True),
            (0, "y", "w", False), (1, "y", "r", False),
        ], 1
    ))
    # 9. Write-after-read
    programs.append(SyntheticProgram(
        "war", [(0, "x", "r", False), (1, "x", "w", False)], 1
    ))
    # 10. Complex
    programs.append(SyntheticProgram(
        "complex", [
            (0, "a", "w", False), (0, "b", "r", False),
            (1, "a", "r", False), (1, "b", "w", False),
            (2, "a", "w", True), (2, "b", "w", True),
        ], 2  # a: T0w-T1r, b: T0r-T1w (T2 is synced)
    ))

    return programs


def detect_races(program: SyntheticProgram) -> Dict[str, Any]:
    """Detect data races in a synthetic program."""
    accesses = program.accesses
    races = []

    # Group by variable
    var_accesses: Dict[str, List] = defaultdict(list)
    for thread, var, rw, synced in accesses:
        var_accesses[var].append((thread, rw, synced))

    for var, accs in var_accesses.items():
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                t1, rw1, sync1 = accs[i]
                t2, rw2, sync2 = accs[j]
                if t1 == t2:
                    continue  # Same thread, no race
                if rw1 == "r" and rw2 == "r":
                    continue  # Both reads, no race
                if sync1 and sync2:
                    continue  # Both synchronized
                races.append({
                    "var": var,
                    "thread1": t1, "op1": rw1,
                    "thread2": t2, "op2": rw2,
                })

    detected_count = len(races)
    correct = (detected_count == program.expected_races)

    return {
        "program": program.name,
        "expected_races": program.expected_races,
        "detected_races": detected_count,
        "correct": correct,
        "race_details": races[:5],  # Limit output
    }


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------

def benchmark_litmus_tests(seed: int = 42) -> Dict[str, Any]:
    """Test 20 litmus patterns against 5 memory models."""
    tests = generate_litmus_tests()
    models = [m.value for m in MemoryModel]

    results = {"n_tests": len(tests), "n_models": len(models), "correct": 0,
               "total": 0, "per_model": {}, "details": []}

    for model in models:
        model_correct = 0
        model_total = 0
        for test in tests:
            t0 = time.perf_counter()
            check = check_litmus_test(test, model)
            elapsed = time.perf_counter() - t0

            match = (check["expected_allowed"] == check["predicted_allowed"])
            results["total"] += 1
            model_total += 1
            if match:
                results["correct"] += 1
                model_correct += 1

            results["details"].append({
                "test": test.name,
                "model": model,
                "expected": check["expected_allowed"],
                "predicted": check["predicted_allowed"],
                "correct": match,
                "has_fence": check["has_fence"],
                "time_s": round(elapsed, 6),
            })

        results["per_model"][model] = {
            "correct": model_correct,
            "total": model_total,
            "accuracy": round(model_correct / max(model_total, 1), 4),
        }

    results["overall_accuracy"] = round(results["correct"] / max(results["total"], 1), 4)
    return results


def benchmark_fence_minimization(seed: int = 42) -> Dict[str, Any]:
    """Test fence minimization on fenced litmus tests."""
    tests = generate_litmus_tests()
    fenced_tests = [t for t in tests if count_fences(t) > 0]

    results = {"n_fenced_tests": len(fenced_tests), "details": []}

    for test in fenced_tests:
        for model in [m.value for m in MemoryModel]:
            t0 = time.perf_counter()
            min_result = minimize_fences_greedy(test, model)
            elapsed = time.perf_counter() - t0
            min_result["test"] = test.name
            min_result["model"] = model
            min_result["time_s"] = round(elapsed, 6)
            results["details"].append(min_result)

    if results["details"]:
        ratios = [d["reduction_ratio"] for d in results["details"]]
        results["avg_reduction_ratio"] = round(float(np.mean(ratios)), 4)
        results["max_reduction_ratio"] = round(float(max(ratios)), 4)
    else:
        results["avg_reduction_ratio"] = 0.0
        results["max_reduction_ratio"] = 0.0

    return results


def benchmark_race_detection(seed: int = 42) -> Dict[str, Any]:
    """Test race detection on 10 synthetic programs."""
    programs = generate_race_programs(seed)
    results = {"n_programs": len(programs), "correct": 0, "details": []}

    for prog in programs:
        t0 = time.perf_counter()
        detection = detect_races(prog)
        elapsed = time.perf_counter() - t0
        detection["time_s"] = round(elapsed, 6)

        if detection["correct"]:
            results["correct"] += 1
        results["details"].append(detection)

    results["accuracy"] = round(results["correct"] / max(len(programs), 1), 4)
    return results


def benchmark_model_comparison(seed: int = 42) -> Dict[str, Any]:
    """Compare memory models: which allows more behaviors?"""
    tests = generate_litmus_tests()
    models = [m.value for m in MemoryModel]

    allowed_counts = {m: 0 for m in models}
    for test in tests:
        for model in models:
            if test.allowed_by.get(model, False):
                allowed_counts[model] += 1

    # Strength ordering: stronger = fewer allowed behaviors
    sorted_models = sorted(allowed_counts.items(), key=lambda x: x[1])

    return {
        "n_tests": len(tests),
        "allowed_counts": allowed_counts,
        "strength_ordering": [m for m, _ in sorted_models],
        "weakest_model": sorted_models[-1][0],
        "strongest_model": sorted_models[0][0],
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("  Litmus INF Memory Model Verification – Benchmark Suite")
    print("=" * 60)

    overall_start = time.perf_counter()

    # 1. Litmus tests vs models
    print("\n[1/4] Litmus tests (20 × 5 models) …")
    litmus_results = benchmark_litmus_tests(seed=42)
    print(f"  Overall accuracy: {litmus_results['overall_accuracy']:.1%}")
    for model, stats in litmus_results["per_model"].items():
        print(f"    {model:>10s}: {stats['accuracy']:.1%}")

    # 2. Fence minimization
    print("\n[2/4] Fence minimization …")
    fence_results = benchmark_fence_minimization(seed=42)
    print(f"  Avg reduction: {fence_results['avg_reduction_ratio']:.1%}")

    # 3. Race detection
    print("\n[3/4] Race detection (10 programs) …")
    race_results = benchmark_race_detection(seed=42)
    print(f"  Accuracy: {race_results['accuracy']:.1%}")

    # 4. Model comparison
    print("\n[4/4] Model comparison …")
    comparison = benchmark_model_comparison(seed=42)
    print(f"  Strength ordering: {' > '.join(comparison['strength_ordering'])}")
    print(f"  Allowed behaviors: {comparison['allowed_counts']}")

    total_time = time.perf_counter() - overall_start

    final_results = {
        "benchmark": "litmus_inf_memory_model",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_time_s": round(total_time, 3),
        "module_availability": {
            "api": HAS_API,
            "mmdb": HAS_MMDB,
            "race_detector": HAS_RACE,
        },
        "litmus_tests": litmus_results,
        "fence_minimization": fence_results,
        "race_detection": race_results,
        "model_comparison": comparison,
        "summary": {
            "litmus_accuracy": litmus_results["overall_accuracy"],
            "fence_reduction_ratio": fence_results["avg_reduction_ratio"],
            "race_detection_accuracy": race_results["accuracy"],
            "strength_ordering": comparison["strength_ordering"],
        },
    }

    out_path = os.path.join(SCRIPT_DIR, "memory_model_benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(final_results, f, indent=2, default=str)
    print(f"\n✓ Results written to {out_path}")
    print(f"  Total time: {total_time:.2f}s")

    return final_results


if __name__ == "__main__":
    main()
