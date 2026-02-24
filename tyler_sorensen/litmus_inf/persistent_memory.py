"""Persistent memory verification: crash consistency, flush/fence analysis, recovery."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re


class PMEMOp(Enum):
    STORE = auto()
    FLUSH = auto()
    FENCE = auto()
    DRAIN = auto()
    PERSIST = auto()
    CLWB = auto()
    CLFLUSH = auto()
    CLFLUSHOPT = auto()
    SFENCE = auto()
    MFENCE = auto()
    NT_STORE = auto()

    def __str__(self) -> str:
        return self.name


class CrashState(Enum):
    CONSISTENT = auto()
    INCONSISTENT = auto()
    PARTIAL_WRITE = auto()
    TORN_WRITE = auto()
    MISSING_DATA = auto()
    STALE_DATA = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class FlushType(Enum):
    CLWB = auto()
    CLFLUSH = auto()
    CLFLUSHOPT = auto()
    NT_STORE_SFENCE = auto()

    def __str__(self) -> str:
        return self.name


class BugSeverity(Enum):
    CRITICAL = auto()
    HIGH = auto()
    MEDIUM = auto()
    LOW = auto()

    def __str__(self) -> str:
        return self.name


# ── Dataclasses ──────────────────────────────────────────────────────────


@dataclass
class PMEMAccess:
    line: int
    op: PMEMOp
    address_expr: str
    size_bytes: int
    in_transaction: bool
    flushed: bool
    fenced: bool

    def __str__(self) -> str:
        flags = []
        if self.in_transaction:
            flags.append("tx")
        if self.flushed:
            flags.append("flushed")
        if self.fenced:
            flags.append("fenced")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        return f"L{self.line}: {self.op} {self.address_expr} ({self.size_bytes}B){flag_str}"


@dataclass
class MissingFlush:
    line: int
    store_line: int
    address_expr: str
    reason: str
    severity: BugSeverity
    fix: str

    def __str__(self) -> str:
        return (f"[{self.severity}] Missing flush at L{self.line} for store at "
                f"L{self.store_line} ({self.address_expr}): {self.reason}")


@dataclass
class OrderingBug:
    line_a: int
    line_b: int
    op_a: str
    op_b: str
    required_order: str
    actual_order: str
    severity: BugSeverity
    fix: str
    explanation: str

    def __str__(self) -> str:
        return (f"[{self.severity}] Ordering bug L{self.line_a}-L{self.line_b}: "
                f"{self.op_a} vs {self.op_b} — {self.explanation}")


@dataclass
class CrashScenario:
    crash_point: int
    pre_crash_ops: List[PMEMAccess]
    expected_state: Dict[str, int]
    actual_state: Dict[str, int]
    consistent: bool

    def __str__(self) -> str:
        status = "CONSISTENT" if self.consistent else "INCONSISTENT"
        return f"Crash@L{self.crash_point}: {status} ({len(self.pre_crash_ops)} ops before crash)"


@dataclass
class CrashConsistencyResult:
    source: str
    scenarios: List[CrashScenario]
    total_crash_points: int
    consistent_count: int
    inconsistent_count: int
    worst_case: Optional[CrashScenario]
    issues: List[str]
    suggestions: List[str]

    def __str__(self) -> str:
        return (f"CrashConsistency: {self.consistent_count}/{self.total_crash_points} consistent, "
                f"{self.inconsistent_count} inconsistent, {len(self.issues)} issues")


@dataclass
class RecoveryResult:
    source: str
    recovery_possible: bool
    recovery_actions: List[str]
    log_based: bool
    undo_possible: bool
    redo_possible: bool
    issues: List[str]

    def __str__(self) -> str:
        status = "recoverable" if self.recovery_possible else "NOT recoverable"
        modes = []
        if self.log_based:
            modes.append("log")
        if self.undo_possible:
            modes.append("undo")
        if self.redo_possible:
            modes.append("redo")
        mode_str = f" ({', '.join(modes)})" if modes else ""
        return f"Recovery: {status}{mode_str}, {len(self.issues)} issues"


@dataclass
class PMEMFenceReport:
    source: str
    stores: List[PMEMAccess]
    flushes: List[PMEMAccess]
    fences: List[PMEMAccess]
    unflushed_stores: List[int]
    unfenced_flushes: List[int]
    ordering_violations: List[OrderingBug]
    estimated_persist_latency_ns: float
    optimized_latency_ns: float

    def __str__(self) -> str:
        return (f"PMEMFence: {len(self.stores)} stores, {len(self.flushes)} flushes, "
                f"{len(self.fences)} fences | unflushed={len(self.unflushed_stores)} "
                f"unfenced={len(self.unfenced_flushes)} | "
                f"latency={self.estimated_persist_latency_ns:.0f}ns "
                f"(optimized={self.optimized_latency_ns:.0f}ns)")


# ── Regex patterns for PMEM operations ───────────────────────────────────

_STORE_PATTERNS = [
    (re.compile(r'\b(?:pmem_memcpy_persist|pmem_memcpy_nodrain)\s*\(([^,]+),'), PMEMOp.PERSIST, 64),
    (re.compile(r'\b_mm_stream_si128\s*\(\s*\(.*?\)\s*([^,)]+)'), PMEMOp.NT_STORE, 16),
    (re.compile(r'\b_mm_stream_si256\s*\(\s*\(.*?\)\s*([^,)]+)'), PMEMOp.NT_STORE, 32),
    (re.compile(r'\b_mm_stream_si64\s*\(\s*([^,)]+)'), PMEMOp.NT_STORE, 8),
    (re.compile(r'\b_mm_stream_pd\s*\(\s*([^,)]+)'), PMEMOp.NT_STORE, 16),
    (re.compile(r'\bpmemobj_tx_add_range\s*\(([^,]+),'), PMEMOp.STORE, 0),
    (re.compile(r'\bpmemobj_persist\s*\([^,]+,\s*([^,]+),'), PMEMOp.PERSIST, 0),
]

_FLUSH_PATTERNS = [
    (re.compile(r'\b_mm_clwb\s*\(\s*([^)]+)\)'), PMEMOp.CLWB),
    (re.compile(r'\b_mm_clflush\s*\(\s*([^)]+)\)'), PMEMOp.CLFLUSH),
    (re.compile(r'\b_mm_clflushopt\s*\(\s*([^)]+)\)'), PMEMOp.CLFLUSHOPT),
    (re.compile(r'\bpmem_flush\s*\(\s*([^,)]+)'), PMEMOp.FLUSH),
    (re.compile(r'\bpmem_persist\s*\(\s*([^,)]+)'), PMEMOp.PERSIST),
    (re.compile(r'\bpmem_msync\s*\(\s*([^,)]+)'), PMEMOp.PERSIST),
]

_FENCE_PATTERNS = [
    (re.compile(r'\b_mm_sfence\s*\('), PMEMOp.SFENCE),
    (re.compile(r'\b_mm_mfence\s*\('), PMEMOp.MFENCE),
    (re.compile(r'\bpmem_drain\s*\('), PMEMOp.DRAIN),
]

_ASSIGN_PATTERN = re.compile(
    r'(?:\*\s*\(([^)]+)\)\s*=|(\w+(?:\.\w+|->?\w+)*)\s*=(?!=))\s*([^;]+);'
)

_TX_BEGIN = re.compile(r'\bpmemobj_tx_begin\s*\(')
_TX_END = re.compile(r'\bpmemobj_tx_(?:end|commit)\s*\(')
_TX_ABORT = re.compile(r'\bpmemobj_tx_abort\s*\(')

_PMEM_INDICATORS = [
    "pmem", "PMEM", "nvm", "NVM", "persistent", "dax", "/dev/dax",
    "pmemobj", "pool", "pop->", "D_RW", "D_RO", "TOID",
]


# ── Helper functions ────────────────────────────────────────────────────


def _is_pmem_region(expr: str) -> bool:
    """Heuristic to detect whether an expression refers to persistent memory."""
    stripped = expr.strip()
    for indicator in _PMEM_INDICATORS:
        if indicator in stripped:
            return True
    if re.search(r'\bD_RW\s*\(', stripped):
        return True
    if re.search(r'\bpmemobj_direct\s*\(', stripped):
        return True
    if stripped.startswith("(") and "void *" in stripped and "pmem" in stripped.lower():
        return True
    if re.search(r'->(?:data|buf|root|hdr|meta|log|entry)', stripped):
        return True
    return False


def _infer_size(expr: str) -> int:
    """Infer write size from expression context."""
    if "uint64" in expr or "long" in expr or "size_t" in expr:
        return 8
    if "uint32" in expr or "int" in expr:
        return 4
    if "uint16" in expr or "short" in expr:
        return 2
    if "uint8" in expr or "char" in expr or "byte" in expr:
        return 1
    if "128" in expr:
        return 16
    if "256" in expr:
        return 32
    return 8  # default pointer-sized


def _check_transaction_boundaries(source: str) -> List[Tuple[int, int]]:
    """Find PMDK transaction boundaries as (start_line, end_line) pairs."""
    boundaries: List[Tuple[int, int]] = []
    lines = source.splitlines()
    tx_stack: List[int] = []
    for i, line in enumerate(lines, 1):
        if _TX_BEGIN.search(line):
            tx_stack.append(i)
        if _TX_END.search(line) or _TX_ABORT.search(line):
            if tx_stack:
                start = tx_stack.pop()
                boundaries.append((start, i))
    return boundaries


def _parse_pmem_ops(source: str) -> List[PMEMAccess]:
    """Parse all PMEM operations from source code."""
    ops: List[PMEMAccess] = []
    lines = source.splitlines()
    tx_bounds = _check_transaction_boundaries(source)

    def _in_tx(line_no: int) -> bool:
        for start, end in tx_bounds:
            if start <= line_no <= end:
                return True
        return False

    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
            continue

        # Check explicit store patterns (PMDK / intrinsics)
        for pattern, op, sz in _STORE_PATTERNS:
            m = pattern.search(stripped)
            if m:
                addr = m.group(1).strip()
                size = sz if sz > 0 else _infer_size(addr)
                is_persist = op == PMEMOp.PERSIST
                ops.append(PMEMAccess(
                    line=i, op=PMEMOp.STORE if not is_persist else PMEMOp.PERSIST,
                    address_expr=addr, size_bytes=size,
                    in_transaction=_in_tx(i),
                    flushed=is_persist or _in_tx(i),
                    fenced=is_persist,
                ))

        # Check flush patterns
        for pattern, op in _FLUSH_PATTERNS:
            m = pattern.search(stripped)
            if m:
                addr = m.group(1).strip()
                is_persist = op == PMEMOp.PERSIST
                ops.append(PMEMAccess(
                    line=i, op=op, address_expr=addr, size_bytes=0,
                    in_transaction=_in_tx(i),
                    flushed=True, fenced=is_persist,
                ))

        # Check fence patterns
        for pattern, op in _FENCE_PATTERNS:
            m = pattern.search(stripped)
            if m:
                ops.append(PMEMAccess(
                    line=i, op=op, address_expr="", size_bytes=0,
                    in_transaction=_in_tx(i),
                    flushed=True, fenced=True,
                ))

        # Check raw pointer stores to PMEM regions
        m = _ASSIGN_PATTERN.search(stripped)
        if m:
            deref_target = m.group(1) or m.group(2)
            if deref_target and _is_pmem_region(deref_target):
                already_covered = any(
                    o.line == i and o.op in (PMEMOp.STORE, PMEMOp.PERSIST, PMEMOp.NT_STORE)
                    for o in ops
                )
                if not already_covered:
                    ops.append(PMEMAccess(
                        line=i, op=PMEMOp.STORE,
                        address_expr=deref_target.strip(),
                        size_bytes=_infer_size(stripped),
                        in_transaction=_in_tx(i),
                        flushed=_in_tx(i),
                        fenced=False,
                    ))

    ops.sort(key=lambda o: o.line)
    return ops


def _trace_flush_coverage(
    stores: List[PMEMAccess], flushes: List[PMEMAccess]
) -> Dict[int, bool]:
    """Determine which store lines are covered by a subsequent flush."""
    coverage: Dict[int, bool] = {}
    flush_addrs: Dict[str, List[int]] = {}
    for f in flushes:
        flush_addrs.setdefault(f.address_expr, []).append(f.line)

    for s in stores:
        covered = False
        if s.in_transaction or s.flushed:
            covered = True
        elif s.op == PMEMOp.PERSIST:
            covered = True
        else:
            addr = s.address_expr
            for flush_addr, flush_lines in flush_addrs.items():
                if _addrs_overlap(addr, flush_addr):
                    if any(fl > s.line for fl in flush_lines):
                        covered = True
                        break
        coverage[s.line] = covered
    return coverage


def _addrs_overlap(store_addr: str, flush_addr: str) -> bool:
    """Heuristic check whether a store address and flush address overlap."""
    s = store_addr.strip().strip("&*() ")
    f = flush_addr.strip().strip("&*() ")
    if s == f:
        return True
    s_base = re.sub(r'\[.*?\]', '', s).split("->")[-1].split(".")[-1]
    f_base = re.sub(r'\[.*?\]', '', f).split("->")[-1].split(".")[-1]
    if s_base and f_base and s_base == f_base:
        return True
    if s in f or f in s:
        return True
    return False


def _enumerate_crash_points(ops: List[PMEMAccess]) -> List[int]:
    """Return all possible crash-point line numbers (after each operation)."""
    if not ops:
        return []
    points: List[int] = []
    for op in ops:
        points.append(op.line)
    last_line = max(o.line for o in ops)
    points.append(last_line + 1)
    return sorted(set(points))


def _compute_persisted_state(
    ops: List[PMEMAccess], crash_point: int
) -> Dict[str, int]:
    """Compute which addresses hold persisted values at a given crash point.

    Only stores that are both flushed and fenced before the crash point
    are guaranteed to be persisted. Stores inside committed transactions
    are treated as atomic (all-or-nothing).
    """
    persisted: Dict[str, int] = {}
    pre_ops = [o for o in ops if o.line < crash_point]

    fenced_before_crash = False
    for o in reversed(pre_ops):
        if o.op in (PMEMOp.SFENCE, PMEMOp.MFENCE, PMEMOp.DRAIN):
            fenced_before_crash = True
            break
        if o.op == PMEMOp.PERSIST:
            fenced_before_crash = True
            break

    # Collect stores and their flush/fence status
    store_ops = [o for o in pre_ops if o.op in (PMEMOp.STORE, PMEMOp.PERSIST, PMEMOp.NT_STORE)]
    flush_ops = [o for o in pre_ops if o.op in (
        PMEMOp.FLUSH, PMEMOp.CLWB, PMEMOp.CLFLUSH, PMEMOp.CLFLUSHOPT, PMEMOp.PERSIST
    )]
    fence_ops = [o for o in pre_ops if o.op in (
        PMEMOp.SFENCE, PMEMOp.MFENCE, PMEMOp.DRAIN, PMEMOp.PERSIST
    )]

    flush_coverage = _trace_flush_coverage(store_ops, flush_ops)

    for idx, s in enumerate(store_ops):
        is_flushed = flush_coverage.get(s.line, False)
        is_fenced = False
        if s.in_transaction:
            # Check if transaction committed before crash
            is_fenced = True
        elif s.op == PMEMOp.PERSIST:
            is_fenced = True
        else:
            for f in fence_ops:
                if f.line > s.line and f.line < crash_point:
                    is_fenced = True
                    break

        if is_flushed and is_fenced:
            persisted[s.address_expr] = s.line

    return persisted


# ── Main analysis functions ──────────────────────────────────────────────


def verify_crash_consistency(source: str) -> CrashConsistencyResult:
    """Verify crash consistency of persistent memory code."""
    ops = _parse_pmem_ops(source)
    store_ops = [o for o in ops if o.op in (PMEMOp.STORE, PMEMOp.PERSIST, PMEMOp.NT_STORE)]

    if not store_ops:
        return CrashConsistencyResult(
            source=source, scenarios=[], total_crash_points=0,
            consistent_count=0, inconsistent_count=0,
            worst_case=None, issues=["No PMEM stores found"], suggestions=[],
        )

    crash_points = _enumerate_crash_points(ops)
    # Expected state: all stores have been persisted
    expected: Dict[str, int] = {s.address_expr: s.line for s in store_ops}

    scenarios: List[CrashScenario] = []
    consistent_count = 0
    inconsistent_count = 0
    worst_case: Optional[CrashScenario] = None
    worst_missing = -1

    for cp in crash_points:
        pre = [o for o in ops if o.line < cp]
        actual = _compute_persisted_state(ops, cp)

        # Consistency: either all stores persisted or none — check for partial
        persisted_addrs = set(actual.keys())
        all_addrs = set(expected.keys())
        stores_before = {s.address_expr for s in store_ops if s.line < cp}

        if not stores_before:
            is_consistent = True  # no stores yet, trivially consistent
        elif stores_before == persisted_addrs:
            is_consistent = True  # all stores before crash are persisted
        elif not persisted_addrs:
            is_consistent = True  # nothing persisted, clean rollback
        else:
            # Partial persistence: check transaction grouping
            tx_bounds = _check_transaction_boundaries(source)
            is_consistent = True
            for start, end in tx_bounds:
                tx_stores = {s.address_expr for s in store_ops
                             if start <= s.line <= end and s.line < cp}
                tx_persisted = tx_stores & persisted_addrs
                if tx_persisted and tx_persisted != tx_stores:
                    is_consistent = False
                    break
            # Non-tx stores: partial is inconsistent if there are dependent writes
            non_tx = stores_before - {s.address_expr for s in store_ops
                                       if any(st <= s.line <= en for st, en in tx_bounds)}
            non_tx_persisted = non_tx & persisted_addrs
            if non_tx_persisted and non_tx_persisted != non_tx and len(non_tx) > 1:
                is_consistent = False

        scenario = CrashScenario(
            crash_point=cp, pre_crash_ops=pre,
            expected_state=expected, actual_state=actual,
            consistent=is_consistent,
        )
        scenarios.append(scenario)

        if is_consistent:
            consistent_count += 1
        else:
            inconsistent_count += 1
            missing = len(stores_before - persisted_addrs)
            if missing > worst_missing:
                worst_missing = missing
                worst_case = scenario

    issues: List[str] = []
    suggestions: List[str] = []
    if inconsistent_count > 0:
        issues.append(f"{inconsistent_count} crash points lead to inconsistent state")
        suggestions.append("Wrap related stores in a PMDK transaction for atomicity")
        suggestions.append("Ensure every store is flushed and fenced before dependent stores")
    unflushed = [s for s in store_ops if not s.flushed and not s.in_transaction]
    if unflushed:
        issues.append(f"{len(unflushed)} stores are never flushed")
        suggestions.append("Add pmem_persist() or clwb+sfence after each store")

    return CrashConsistencyResult(
        source=source, scenarios=scenarios,
        total_crash_points=len(crash_points),
        consistent_count=consistent_count,
        inconsistent_count=inconsistent_count,
        worst_case=worst_case, issues=issues, suggestions=suggestions,
    )


def detect_missing_flushes(source: str) -> List[MissingFlush]:
    """Detect stores to PMEM that lack a corresponding flush."""
    ops = _parse_pmem_ops(source)
    store_ops = [o for o in ops if o.op in (PMEMOp.STORE, PMEMOp.NT_STORE)]
    flush_ops = [o for o in ops if o.op in (
        PMEMOp.FLUSH, PMEMOp.CLWB, PMEMOp.CLFLUSH, PMEMOp.CLFLUSHOPT, PMEMOp.PERSIST
    )]
    fence_lines = sorted(
        o.line for o in ops
        if o.op in (PMEMOp.SFENCE, PMEMOp.MFENCE, PMEMOp.DRAIN, PMEMOp.PERSIST)
    )

    coverage = _trace_flush_coverage(store_ops, flush_ops)
    missing: List[MissingFlush] = []

    for s in store_ops:
        if s.in_transaction:
            continue  # PMDK tx auto-flushes
        if s.op == PMEMOp.PERSIST:
            continue
        if coverage.get(s.line, False):
            continue

        # Find next fence/persist point
        next_fence = None
        for fl in fence_lines:
            if fl > s.line:
                next_fence = fl
                break

        if s.op == PMEMOp.NT_STORE:
            reason = "Non-temporal store requires sfence to guarantee persistence"
            severity = BugSeverity.HIGH
            fix = f"Add _mm_sfence() after NT store at line {s.line}"
        elif next_fence is None:
            reason = "Store has no flush and no subsequent fence/persist point"
            severity = BugSeverity.CRITICAL
            fix = f"Add pmem_persist({s.address_expr}, {s.size_bytes}) after line {s.line}"
        else:
            reason = f"Store is not flushed before fence at line {next_fence}"
            severity = BugSeverity.HIGH
            fix = (f"Add _mm_clwb({s.address_expr}) between "
                   f"line {s.line} and line {next_fence}")

        missing.append(MissingFlush(
            line=s.line, store_line=s.line,
            address_expr=s.address_expr,
            reason=reason, severity=severity, fix=fix,
        ))

    return missing


def detect_ordering_bugs(source: str) -> List[OrderingBug]:
    """Detect PMEM store/flush/fence ordering violations."""
    ops = _parse_pmem_ops(source)
    bugs: List[OrderingBug] = []

    store_ops = [o for o in ops if o.op in (PMEMOp.STORE, PMEMOp.NT_STORE)]
    flush_ops = [o for o in ops if o.op in (
        PMEMOp.FLUSH, PMEMOp.CLWB, PMEMOp.CLFLUSHOPT, PMEMOp.PERSIST
    )]
    fence_ops = [o for o in ops if o.op in (PMEMOp.SFENCE, PMEMOp.MFENCE, PMEMOp.DRAIN)]
    clflush_ops = [o for o in ops if o.op == PMEMOp.CLFLUSH]

    # 1) clwb/clflushopt without subsequent sfence
    for f in flush_ops:
        if f.op not in (PMEMOp.CLWB, PMEMOp.CLFLUSHOPT):
            continue
        has_fence = any(fe.line > f.line for fe in fence_ops)
        if not has_fence:
            bugs.append(OrderingBug(
                line_a=f.line, line_b=f.line,
                op_a=str(f.op), op_b="(missing sfence)",
                required_order=f"{f.op} → sfence",
                actual_order=f"{f.op} with no subsequent sfence",
                severity=BugSeverity.CRITICAL,
                fix=f"Add _mm_sfence() after {f.op} at line {f.line}",
                explanation=(f"{f.op} only initiates cache line writeback; "
                             "sfence is required to order subsequent operations"),
            ))

    # 2) Store after flush without intervening fence (store reordered before flush completes)
    for s in store_ops:
        if s.in_transaction:
            continue
        preceding_flushes = [f for f in flush_ops if f.line < s.line]
        for pf in preceding_flushes:
            if _addrs_overlap(s.address_expr, pf.address_expr):
                continue  # same address, not an ordering issue
            fence_between = any(
                fe.line > pf.line and fe.line < s.line for fe in fence_ops
            )
            if not fence_between and pf.op != PMEMOp.CLFLUSH:
                bugs.append(OrderingBug(
                    line_a=pf.line, line_b=s.line,
                    op_a=f"flush({pf.address_expr})",
                    op_b=f"store({s.address_expr})",
                    required_order="flush → fence → store",
                    actual_order="flush → store (no fence)",
                    severity=BugSeverity.HIGH,
                    fix=f"Add _mm_sfence() between line {pf.line} and line {s.line}",
                    explanation="Without a fence, the store may be reordered before "
                                "the flush completes, risking data loss on crash",
                ))

    # 3) Out-of-order persist: store A before store B but B could persist first
    for i, sa in enumerate(store_ops):
        if sa.in_transaction:
            continue
        for sb in store_ops[i + 1:]:
            if sb.in_transaction:
                continue
            if sa.address_expr == sb.address_expr:
                continue
            # Check if A is flushed+fenced before B is stored
            a_fenced = False
            for fe in fence_ops:
                if fe.line > sa.line and fe.line < sb.line:
                    # Check if A was flushed before this fence
                    a_flushed_before_fence = any(
                        f.line > sa.line and f.line <= fe.line
                        and _addrs_overlap(sa.address_expr, f.address_expr)
                        for f in flush_ops
                    )
                    if a_flushed_before_fence:
                        a_fenced = True
                        break
            if not a_fenced and not sa.flushed:
                bugs.append(OrderingBug(
                    line_a=sa.line, line_b=sb.line,
                    op_a=f"store({sa.address_expr})",
                    op_b=f"store({sb.address_expr})",
                    required_order="store A → flush A → fence → store B",
                    actual_order="store A → store B (A not persist-ordered before B)",
                    severity=BugSeverity.MEDIUM,
                    fix=(f"Add flush+fence for {sa.address_expr} between "
                         f"line {sa.line} and line {sb.line}"),
                    explanation="Store B may persist before store A on crash, "
                                "leading to inconsistent state",
                ))

    # 4) PMDK transaction ordering: check for stores outside tx that depend on tx
    tx_bounds = _check_transaction_boundaries(source)
    for s in store_ops:
        if s.in_transaction:
            continue
        for start, end in tx_bounds:
            if s.line > end:
                fence_after_tx = any(
                    fe.line > end and fe.line <= s.line for fe in fence_ops
                )
                if not fence_after_tx:
                    has_persist = any(
                        o.op == PMEMOp.PERSIST and o.line > end and o.line <= s.line
                        for o in ops
                    )
                    if not has_persist:
                        bugs.append(OrderingBug(
                            line_a=end, line_b=s.line,
                            op_a=f"tx_end(L{start}-L{end})",
                            op_b=f"store({s.address_expr})",
                            required_order="tx_commit → fence → dependent store",
                            actual_order="tx_commit → store (no fence)",
                            severity=BugSeverity.HIGH,
                            fix=f"Add pmem_drain() between line {end} and line {s.line}",
                            explanation="Store after transaction may see uncommitted "
                                        "state if transaction recovery rolls back",
                        ))

    return bugs


def verify_recovery(source: str) -> RecoveryResult:
    """Check for crash recovery logic in persistent memory code."""
    issues: List[str] = []
    actions: List[str] = []
    log_based = False
    undo_possible = False
    redo_possible = False

    lines = source.splitlines()
    has_recovery_func = bool(re.search(
        r'\b(?:recover|recovery|replay|restore|rollback|redo|undo)\s*\(', source, re.IGNORECASE
    ))
    has_wal = bool(re.search(r'\b(?:wal|write.ahead.log|WAL)\b', source, re.IGNORECASE))
    has_undo_log = bool(re.search(r'\b(?:undo.log|undo_log|UNDO)\b', source, re.IGNORECASE))
    has_redo_log = bool(re.search(r'\b(?:redo.log|redo_log|REDO)\b', source, re.IGNORECASE))
    has_pmdk_tx = bool(_TX_BEGIN.search(source))
    has_check_pool = bool(re.search(r'\bpmemobj_check\s*\(', source))
    has_open_pool = bool(re.search(r'\bpmemobj_open\s*\(', source))
    has_valid_check = bool(re.search(
        r'\b(?:is_valid|validate|consistency_check|check_consistency|verify)\s*\(', source, re.IGNORECASE
    ))

    if has_wal or has_undo_log or has_redo_log:
        log_based = True
        if has_undo_log:
            undo_possible = True
            actions.append("Undo log detected: can rollback incomplete operations")
        if has_redo_log:
            redo_possible = True
            actions.append("Redo log detected: can replay committed operations")
        if has_wal:
            undo_possible = True
            redo_possible = True
            actions.append("WAL detected: supports both undo and redo recovery")

    if has_pmdk_tx:
        log_based = True
        undo_possible = True
        actions.append("PMDK transactions provide automatic undo-log recovery")
        if not has_check_pool and not has_open_pool:
            issues.append("PMDK transactions found but no pool open/check — "
                          "recovery may not trigger on restart")

    if has_recovery_func:
        actions.append("Explicit recovery function detected")
    elif not has_pmdk_tx and not log_based:
        issues.append("No recovery mechanism found — data may be lost on crash")

    if has_valid_check:
        actions.append("Consistency validation function detected")
    elif has_recovery_func or log_based:
        issues.append("Recovery exists but no post-recovery validation found")

    # Check if recovery covers all crash scenarios
    ops = _parse_pmem_ops(source)
    store_count = sum(1 for o in ops if o.op in (PMEMOp.STORE, PMEMOp.NT_STORE))
    tx_bounds = _check_transaction_boundaries(source)
    stores_in_tx = sum(
        1 for o in ops if o.op == PMEMOp.STORE
        and any(s <= o.line <= e for s, e in tx_bounds)
    )
    stores_outside_tx = store_count - stores_in_tx

    if stores_outside_tx > 0 and not has_recovery_func and not has_wal:
        issues.append(f"{stores_outside_tx} stores outside transactions lack recovery coverage")

    recovery_possible = has_recovery_func or has_pmdk_tx or log_based

    return RecoveryResult(
        source=source,
        recovery_possible=recovery_possible,
        recovery_actions=actions,
        log_based=log_based,
        undo_possible=undo_possible,
        redo_possible=redo_possible,
        issues=issues,
    )


def pmem_fence_analysis(source: str) -> PMEMFenceReport:
    """Full analysis of PMEM fence usage with latency estimation."""
    ops = _parse_pmem_ops(source)

    stores = [o for o in ops if o.op in (PMEMOp.STORE, PMEMOp.NT_STORE, PMEMOp.PERSIST)]
    flushes = [o for o in ops if o.op in (
        PMEMOp.FLUSH, PMEMOp.CLWB, PMEMOp.CLFLUSH, PMEMOp.CLFLUSHOPT, PMEMOp.PERSIST
    )]
    fences = [o for o in ops if o.op in (PMEMOp.SFENCE, PMEMOp.MFENCE, PMEMOp.DRAIN)]

    # Unflushed stores
    coverage = _trace_flush_coverage(stores, flushes)
    unflushed = [s.line for s in stores if not coverage.get(s.line, False)]

    # Unfenced flushes: flushes that have no subsequent fence
    unfenced: List[int] = []
    fence_lines = sorted(f.line for f in fences)
    for f in flushes:
        if f.op in (PMEMOp.PERSIST, PMEMOp.CLFLUSH):
            continue  # persist includes fence; clflush is ordered
        has_fence_after = any(fl > f.line for fl in fence_lines)
        if not has_fence_after:
            unfenced.append(f.line)

    ordering_violations = detect_ordering_bugs(source)

    # Latency estimation (nanoseconds)
    LATENCY = {
        PMEMOp.CLWB: 20.0,
        PMEMOp.CLFLUSH: 100.0,
        PMEMOp.CLFLUSHOPT: 20.0,
        PMEMOp.FLUSH: 20.0,
        PMEMOp.PERSIST: 120.0,  # flush + drain combined
        PMEMOp.SFENCE: 5.0,
        PMEMOp.MFENCE: 30.0,
        PMEMOp.DRAIN: 5.0,
        PMEMOp.STORE: 0.0,
        PMEMOp.NT_STORE: 0.0,
    }

    estimated_ns = 0.0
    for o in ops:
        estimated_ns += LATENCY.get(o.op, 0.0)

    # Optimized latency: replace clflush with clflushopt+sfence batching
    optimized_ns = 0.0
    clflush_count = sum(1 for o in flushes if o.op == PMEMOp.CLFLUSH)
    non_clflush = [o for o in ops if o.op != PMEMOp.CLFLUSH]

    if clflush_count > 0:
        # Replace N clflush with N clflushopt + 1 sfence
        for o in non_clflush:
            optimized_ns += LATENCY.get(o.op, 0.0)
        optimized_ns += clflush_count * LATENCY[PMEMOp.CLFLUSHOPT]
        optimized_ns += LATENCY[PMEMOp.SFENCE]  # single sfence for the batch
    else:
        optimized_ns = estimated_ns

    # Coalesce adjacent flushes to same cache line
    if len(flushes) > 1:
        unique_addrs: Set[str] = set()
        redundant_count = 0
        for f in flushes:
            base = re.sub(r'\[.*?\]', '', f.address_expr).strip()
            if base in unique_addrs:
                redundant_count += 1
            else:
                unique_addrs.add(base)
        if redundant_count > 0:
            optimized_ns -= redundant_count * LATENCY.get(PMEMOp.CLWB, 20.0)
            optimized_ns = max(optimized_ns, 0.0)

    return PMEMFenceReport(
        source=source,
        stores=stores, flushes=flushes, fences=fences,
        unflushed_stores=unflushed,
        unfenced_flushes=unfenced,
        ordering_violations=ordering_violations,
        estimated_persist_latency_ns=estimated_ns,
        optimized_latency_ns=optimized_ns,
    )
