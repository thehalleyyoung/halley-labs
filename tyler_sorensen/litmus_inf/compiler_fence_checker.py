"""Compiler fence checker: analyzes C/LLVM-IR/Rust source for missing,
redundant, or incorrectly-typed memory barriers across architectures."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CompilerType(Enum):
    GCC = auto()
    LLVM = auto()
    RUSTC = auto()
    MSVC = auto()

    def __str__(self) -> str:
        return self.name


class FenceKind(Enum):
    FULL_BARRIER = auto()
    STORE_BARRIER = auto()
    LOAD_BARRIER = auto()
    ACQUIRE_BARRIER = auto()
    RELEASE_BARRIER = auto()
    COMPILER_BARRIER = auto()
    ATOMIC_FENCE = auto()
    INTRINSIC_BARRIER = auto()

    def __str__(self) -> str:
        return self.name.replace("_", " ").title()


class BarrierStatus(Enum):
    CORRECT = auto()
    MISSING = auto()
    REDUNDANT = auto()
    WRONG_TYPE = auto()
    MISPLACED = auto()

    def __str__(self) -> str:
        return self.name


class OptLevel(Enum):
    O0 = "O0"
    O1 = "O1"
    O2 = "O2"
    O3 = "O3"
    Os = "Os"
    Oz = "Oz"

    def __str__(self) -> str:
        return self.value


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class FenceLocation:
    line: int
    column: int
    fence_kind: FenceKind
    source_text: str
    in_function: Optional[str]
    in_loop: bool

    def __str__(self) -> str:
        loc = f"L{self.line}:C{self.column}"
        fn = f" in {self.in_function}()" if self.in_function else ""
        loop = " [loop]" if self.in_loop else ""
        return f"{self.fence_kind} at {loc}{fn}{loop}: {self.source_text.strip()}"


@dataclass
class MissingBarrier:
    line: int
    between_ops: Tuple[str, str]
    required_fence: FenceKind
    reason: str
    severity: int
    fix_suggestion: str

    def __str__(self) -> str:
        return (f"MISSING {self.required_fence} at L{self.line} "
                f"between {self.between_ops[0]} and {self.between_ops[1]} "
                f"(severity={self.severity}): {self.reason}")


@dataclass
class RedundantFence:
    line: int
    fence_kind: FenceKind
    reason: str
    can_remove: bool
    savings_cycles: float

    def __str__(self) -> str:
        tag = "removable" if self.can_remove else "kept"
        return (f"REDUNDANT {self.fence_kind} at L{self.line} [{tag}] "
                f"saves ~{self.savings_cycles:.1f} cycles: {self.reason}")


@dataclass
class FenceCheckResult:
    source_file: str
    compiler: CompilerType
    target_arch: str
    fences_found: List[FenceLocation] = field(default_factory=list)
    missing: List[MissingBarrier] = field(default_factory=list)
    redundant: List[RedundantFence] = field(default_factory=list)
    total_fence_cost: float = 0.0
    optimized_fence_cost: float = 0.0
    score: float = 100.0

    def __str__(self) -> str:
        return (f"FenceCheck[{self.compiler} / {self.target_arch}]: "
                f"{len(self.fences_found)} fences, "
                f"{len(self.missing)} missing, "
                f"{len(self.redundant)} redundant, "
                f"cost={self.total_fence_cost:.1f} → "
                f"{self.optimized_fence_cost:.1f}, score={self.score:.1f}")


@dataclass
class OptimizationResult:
    original_source: str
    optimized_source: str
    fences_removed: int
    fences_added: int
    fences_replaced: int
    estimated_speedup_pct: float
    correctness_preserved: bool

    def __str__(self) -> str:
        return (f"Optimization: -{self.fences_removed}/+{self.fences_added}/"
                f"~{self.fences_replaced} fences, "
                f"speedup≈{self.estimated_speedup_pct:.1f}%, "
                f"correct={self.correctness_preserved}")


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_ORDERING_STRENGTH: Dict[str, int] = {
    "relaxed": 0, "monotonic": 0,
    "consume": 1,
    "acquire": 2,
    "release": 2,
    "acq_rel": 3,
    "seq_cst": 4,
}

_FENCE_COST_TABLE: Dict[Tuple[FenceKind, str], float] = {
    (FenceKind.FULL_BARRIER, "x86_64"): 33.0,
    (FenceKind.FULL_BARRIER, "aarch64"): 40.0,
    (FenceKind.FULL_BARRIER, "riscv64"): 35.0,
    (FenceKind.FULL_BARRIER, "ppc64"): 45.0,
    (FenceKind.STORE_BARRIER, "x86_64"): 8.0,
    (FenceKind.STORE_BARRIER, "aarch64"): 15.0,
    (FenceKind.STORE_BARRIER, "riscv64"): 12.0,
    (FenceKind.STORE_BARRIER, "ppc64"): 18.0,
    (FenceKind.LOAD_BARRIER, "x86_64"): 0.0,
    (FenceKind.LOAD_BARRIER, "aarch64"): 12.0,
    (FenceKind.LOAD_BARRIER, "riscv64"): 10.0,
    (FenceKind.LOAD_BARRIER, "ppc64"): 14.0,
    (FenceKind.ACQUIRE_BARRIER, "x86_64"): 0.0,
    (FenceKind.ACQUIRE_BARRIER, "aarch64"): 13.0,
    (FenceKind.ACQUIRE_BARRIER, "riscv64"): 11.0,
    (FenceKind.ACQUIRE_BARRIER, "ppc64"): 15.0,
    (FenceKind.RELEASE_BARRIER, "x86_64"): 0.0,
    (FenceKind.RELEASE_BARRIER, "aarch64"): 13.0,
    (FenceKind.RELEASE_BARRIER, "riscv64"): 11.0,
    (FenceKind.RELEASE_BARRIER, "ppc64"): 15.0,
    (FenceKind.COMPILER_BARRIER, "x86_64"): 0.0,
    (FenceKind.COMPILER_BARRIER, "aarch64"): 0.0,
    (FenceKind.COMPILER_BARRIER, "riscv64"): 0.0,
    (FenceKind.COMPILER_BARRIER, "ppc64"): 0.0,
    (FenceKind.ATOMIC_FENCE, "x86_64"): 20.0,
    (FenceKind.ATOMIC_FENCE, "aarch64"): 30.0,
    (FenceKind.ATOMIC_FENCE, "riscv64"): 25.0,
    (FenceKind.ATOMIC_FENCE, "ppc64"): 35.0,
    (FenceKind.INTRINSIC_BARRIER, "x86_64"): 25.0,
    (FenceKind.INTRINSIC_BARRIER, "aarch64"): 32.0,
    (FenceKind.INTRINSIC_BARRIER, "riscv64"): 28.0,
    (FenceKind.INTRINSIC_BARRIER, "ppc64"): 38.0,
}

_TSO_ARCHS: Set[str] = {"x86_64", "x86", "i686", "amd64", "sparc"}

_WEAK_ARCHS: Set[str] = {"aarch64", "arm", "arm64", "riscv64", "riscv32",
                          "ppc64", "ppc", "mips", "mips64", "alpha"}

_THREAD_FUNC_PATTERNS = re.compile(
    r'\b(?:thread_|worker_|producer_|consumer_|sender_|receiver_|'
    r'task_|run_|do_work|parallel_)\w*', re.IGNORECASE)

_GCC_ORDERING_MAP: Dict[str, str] = {
    "__ATOMIC_RELAXED": "relaxed",
    "__ATOMIC_CONSUME": "consume",
    "__ATOMIC_ACQUIRE": "acquire",
    "__ATOMIC_RELEASE": "release",
    "__ATOMIC_ACQ_REL": "acq_rel",
    "__ATOMIC_SEQ_CST": "seq_cst",
}

_RUST_ORDERING_MAP: Dict[str, str] = {
    "Relaxed": "relaxed",
    "Acquire": "acquire",
    "Release": "release",
    "AcqRel": "acq_rel",
    "SeqCst": "seq_cst",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _fence_cost(fence_kind: FenceKind, arch: str) -> float:
    """Return estimated cycle cost for a fence on the given architecture."""
    norm = arch.lower().replace("-", "").replace("_", "")
    for key_arch in ("x86_64", "aarch64", "riscv64", "ppc64"):
        if key_arch in norm or norm in key_arch:
            return _FENCE_COST_TABLE.get((fence_kind, key_arch), 20.0)
    return _FENCE_COST_TABLE.get((fence_kind, "aarch64"), 20.0)


def _ordering_sufficient(ordering: str, required: str) -> bool:
    """Return True when *ordering* is at least as strong as *required*."""
    o = ordering.lower().replace("::", "").strip()
    r = required.lower().replace("::", "").strip()
    return _ORDERING_STRENGTH.get(o, -1) >= _ORDERING_STRENGTH.get(r, 5)


def _is_tso(arch: str) -> bool:
    norm = arch.lower().replace("-", "").replace("_", "")
    return any(t in norm for t in _TSO_ARCHS)


def _is_weak(arch: str) -> bool:
    norm = arch.lower().replace("-", "").replace("_", "")
    return any(w in norm for w in _WEAK_ARCHS)


def _current_function(source: str, line_idx: int) -> Optional[str]:
    """Walk backwards from *line_idx* to find the enclosing function name."""
    lines = source.splitlines()
    brace_depth = 0
    for i in range(line_idx, -1, -1):
        brace_depth += lines[i].count('}') - lines[i].count('{')
        m = re.match(r'(?:void|int|bool|unsigned|static\s+\w+)\s+(\w+)\s*\(', lines[i])
        if m and brace_depth <= 0:
            return m.group(1)
        m_rust = re.match(r'\s*(?:pub\s+)?(?:unsafe\s+)?fn\s+(\w+)', lines[i])
        if m_rust and brace_depth <= 0:
            return m_rust.group(1)
    return None


def _in_loop(source: str, line_idx: int) -> bool:
    """Heuristic: return True if *line_idx* is inside a loop construct."""
    lines = source.splitlines()
    brace_depth = 0
    for i in range(line_idx, -1, -1):
        brace_depth += lines[i].count('}') - lines[i].count('{')
        if re.match(r'\s*(for|while|do)\b', lines[i]) and brace_depth <= 0:
            return True
    return False


def _extract_column(line_text: str, pattern: str) -> int:
    m = re.search(pattern, line_text)
    return m.start() + 1 if m else 1


def _detect_access_pairs(source: str) -> List[Tuple[int, str, int, str, Optional[str]]]:
    """Find pairs of memory accesses that might need ordering constraints.

    Returns list of (line1, op1, line2, op2, enclosing_function).
    A pair is (store, load), (store, store), or (load, store) on potentially
    shared variables within the same function body.
    """
    store_pat = re.compile(
        r'(?:__atomic_store|__atomic_exchange|__sync_lock_release|'
        r'\.store\s*\(|store\s+atomic|atomicrmw|__atomic_fetch_\w+|'
        r'\.fetch_\w+\s*\()', re.IGNORECASE)
    load_pat = re.compile(
        r'(?:__atomic_load|__sync_lock_test_and_set|'
        r'\.load\s*\(|load\s+atomic|__atomic_compare_exchange|'
        r'\.compare_exchange)', re.IGNORECASE)

    lines = source.splitlines()
    accesses: List[Tuple[int, str]] = []  # (line_no, "store"|"load")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        if store_pat.search(line):
            accesses.append((idx + 1, "store"))
        if load_pat.search(line):
            accesses.append((idx + 1, "load"))

    pairs: List[Tuple[int, str, int, str, Optional[str]]] = []
    for i in range(len(accesses) - 1):
        l1, op1 = accesses[i]
        l2, op2 = accesses[i + 1]
        if l2 - l1 > 50:
            continue
        fn1 = _current_function(source, l1 - 1)
        fn2 = _current_function(source, l2 - 1)
        if fn1 == fn2:
            pairs.append((l1, op1, l2, op2, fn1))
    return pairs


def _gcc_fence_kind(asm_text: str) -> FenceKind:
    """Map an inline-asm barrier string to a FenceKind."""
    text = asm_text.lower()
    if "mfence" in text:
        return FenceKind.FULL_BARRIER
    if "sfence" in text:
        return FenceKind.STORE_BARRIER
    if "lfence" in text:
        return FenceKind.LOAD_BARRIER
    if "dmb ish" in text:
        return FenceKind.FULL_BARRIER
    if "dmb ishst" in text:
        return FenceKind.STORE_BARRIER
    if "dmb ishld" in text:
        return FenceKind.LOAD_BARRIER
    if "fence rw,rw" in text or "fence iorw,iorw" in text:
        return FenceKind.FULL_BARRIER
    if "fence r,r" in text:
        return FenceKind.LOAD_BARRIER
    if "fence w,w" in text:
        return FenceKind.STORE_BARRIER
    if "fence r,rw" in text:
        return FenceKind.ACQUIRE_BARRIER
    if "fence rw,w" in text:
        return FenceKind.RELEASE_BARRIER
    if '""' in text or "memory" in text:
        return FenceKind.COMPILER_BARRIER
    return FenceKind.FULL_BARRIER


def _gcc_ordering_to_fence(ordering_str: str) -> FenceKind:
    canon = _GCC_ORDERING_MAP.get(ordering_str, ordering_str).lower()
    mapping: Dict[str, FenceKind] = {
        "relaxed": FenceKind.COMPILER_BARRIER,
        "consume": FenceKind.ACQUIRE_BARRIER,
        "acquire": FenceKind.ACQUIRE_BARRIER,
        "release": FenceKind.RELEASE_BARRIER,
        "acq_rel": FenceKind.FULL_BARRIER,
        "seq_cst": FenceKind.FULL_BARRIER,
    }
    return mapping.get(canon, FenceKind.FULL_BARRIER)


def _llvm_ordering_to_fence(ordering: str) -> FenceKind:
    mapping: Dict[str, FenceKind] = {
        "monotonic": FenceKind.COMPILER_BARRIER,
        "acquire": FenceKind.ACQUIRE_BARRIER,
        "release": FenceKind.RELEASE_BARRIER,
        "acq_rel": FenceKind.FULL_BARRIER,
        "seq_cst": FenceKind.FULL_BARRIER,
    }
    return mapping.get(ordering.lower().strip(), FenceKind.FULL_BARRIER)


def _rust_ordering_to_fence(ordering: str) -> FenceKind:
    canon = _RUST_ORDERING_MAP.get(ordering.strip(), ordering.strip().lower())
    mapping: Dict[str, FenceKind] = {
        "relaxed": FenceKind.COMPILER_BARRIER,
        "acquire": FenceKind.ACQUIRE_BARRIER,
        "release": FenceKind.RELEASE_BARRIER,
        "acq_rel": FenceKind.FULL_BARRIER,
        "seq_cst": FenceKind.FULL_BARRIER,
    }
    return mapping.get(canon, FenceKind.FULL_BARRIER)


def _required_ordering_for_pair(op1: str, op2: str, arch: str) -> str:
    """Return the minimum ordering required between two access types."""
    if _is_tso(arch):
        if op1 == "store" and op2 == "load":
            return "seq_cst"
        return "relaxed"
    if op1 == "store" and op2 == "load":
        return "acq_rel"
    if op1 == "store" and op2 == "store":
        return "release"
    if op1 == "load" and op2 == "store":
        return "acquire"
    if op1 == "load" and op2 == "load":
        return "acquire"
    return "acq_rel"


def _has_fence_between(source: str, line1: int, line2: int) -> bool:
    """Return True if any barrier exists between *line1* and *line2*."""
    fence_pat = re.compile(
        r'(?:__sync_synchronize|asm\s+volatile|__atomic_thread_fence|'
        r'fence\s*\(|compiler_fence|atomic::fence|'
        r'std::atomic_thread_fence|_mm_mfence|_mm_sfence|_mm_lfence|'
        r'MemoryBarrier|_ReadWriteBarrier)',
        re.IGNORECASE)
    lines = source.splitlines()
    for i in range(line1, min(line2, len(lines))):
        if fence_pat.search(lines[i]):
            return True
    return False


def _extract_ordering_from_line(line_text: str) -> Optional[str]:
    """Pull out a GCC / Rust / LLVM ordering token from a source line."""
    for gcc_tok, canon in _GCC_ORDERING_MAP.items():
        if gcc_tok in line_text:
            return canon
    m = re.search(r'Ordering::(\w+)', line_text)
    if m:
        return _RUST_ORDERING_MAP.get(m.group(1), m.group(1).lower())
    for llvm_ord in ("seq_cst", "acq_rel", "release", "acquire", "monotonic"):
        if llvm_ord in line_text.lower():
            return llvm_ord
    return None


# ---------------------------------------------------------------------------
# Public API — GCC
# ---------------------------------------------------------------------------

def check_gcc_fences(c_source: str, target_arch: str = "x86_64") -> FenceCheckResult:
    """Analyze C source compiled with GCC for fence correctness."""
    lines = c_source.splitlines()
    fences: List[FenceLocation] = []
    result = FenceCheckResult(
        source_file="<stdin>", compiler=CompilerType.GCC,
        target_arch=target_arch,
    )

    # --- patterns ---
    pat_atomic_load = re.compile(r'__atomic_load\w*\s*\(')
    pat_atomic_store = re.compile(r'__atomic_store\w*\s*\(')
    pat_atomic_exchange = re.compile(r'__atomic_exchange\w*\s*\(')
    pat_atomic_cmpxchg = re.compile(r'__atomic_compare_exchange\w*\s*\(')
    pat_atomic_fetch = re.compile(r'__atomic_fetch_(?:add|sub|and|or|xor|nand)\s*\(')
    pat_sync_sync = re.compile(r'__sync_synchronize\s*\(')
    pat_sync_lock = re.compile(r'__sync_lock_test_and_set\s*\(')
    pat_sync_lock_rel = re.compile(r'__sync_lock_release\s*\(')
    pat_sync_val = re.compile(r'__sync_val_compare_and_swap\s*\(')
    pat_sync_fetch = re.compile(r'__sync_fetch_and_(?:add|sub|and|or|xor|nand)\s*\(')
    pat_asm_barrier = re.compile(
        r'(?:asm|__asm__)\s+(?:volatile|__volatile__)\s*\(\s*"([^"]*)"')
    pat_builtin_ia32 = re.compile(r'__builtin_ia32_(?:mfence|sfence|lfence)\s*\(')
    pat_thread_fence = re.compile(r'__atomic_thread_fence\s*\(')
    pat_signal_fence = re.compile(r'__atomic_signal_fence\s*\(')

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        lineno = idx + 1
        fn_name = _current_function(c_source, idx)
        loop = _in_loop(c_source, idx)

        if pat_sync_sync.search(line):
            col = _extract_column(line, r'__sync_synchronize')
            fences.append(FenceLocation(lineno, col, FenceKind.FULL_BARRIER,
                                        stripped, fn_name, loop))
        if pat_asm_barrier.search(line):
            m = pat_asm_barrier.search(line)
            asm_text = m.group(1) if m else ""
            kind = _gcc_fence_kind(asm_text)
            col = _extract_column(line, r'asm|__asm__')
            fences.append(FenceLocation(lineno, col, kind, stripped,
                                        fn_name, loop))
        if pat_builtin_ia32.search(line):
            if "mfence" in line:
                kind = FenceKind.FULL_BARRIER
            elif "sfence" in line:
                kind = FenceKind.STORE_BARRIER
            else:
                kind = FenceKind.LOAD_BARRIER
            col = _extract_column(line, r'__builtin_ia32_')
            fences.append(FenceLocation(lineno, col, FenceKind.INTRINSIC_BARRIER,
                                        stripped, fn_name, loop))
        if pat_thread_fence.search(line):
            ordering = _extract_ordering_from_line(line)
            kind = _gcc_ordering_to_fence(ordering or "seq_cst")
            col = _extract_column(line, r'__atomic_thread_fence')
            fences.append(FenceLocation(lineno, col, FenceKind.ATOMIC_FENCE,
                                        stripped, fn_name, loop))
        if pat_signal_fence.search(line):
            col = _extract_column(line, r'__atomic_signal_fence')
            fences.append(FenceLocation(lineno, col, FenceKind.COMPILER_BARRIER,
                                        stripped, fn_name, loop))

        for pat in (pat_atomic_load, pat_atomic_store, pat_atomic_exchange,
                    pat_atomic_cmpxchg, pat_atomic_fetch):
            if pat.search(line):
                ordering = _extract_ordering_from_line(line)
                if ordering:
                    kind = _gcc_ordering_to_fence(ordering)
                    col = _extract_column(line, r'__atomic_')
                    fences.append(FenceLocation(lineno, col, kind, stripped,
                                                fn_name, loop))

        for pat in (pat_sync_lock, pat_sync_lock_rel, pat_sync_val,
                    pat_sync_fetch):
            if pat.search(line):
                col = _extract_column(line, r'__sync_')
                fences.append(FenceLocation(lineno, col, FenceKind.FULL_BARRIER,
                                            stripped, fn_name, loop))

    result.fences_found = fences
    result.missing = missing_compiler_barriers(c_source, target_arch)
    result.redundant = redundant_compiler_fences(c_source, target_arch)
    result.total_fence_cost = sum(_fence_cost(f.fence_kind, target_arch)
                                  for f in fences)
    removable = sum(r.savings_cycles for r in result.redundant if r.can_remove)
    result.optimized_fence_cost = max(0.0, result.total_fence_cost - removable)
    n_issues = len(result.missing) * 3 + len(result.redundant)
    result.score = max(0.0, 100.0 - n_issues * 5.0)
    return result


# ---------------------------------------------------------------------------
# Public API — LLVM IR
# ---------------------------------------------------------------------------

def check_llvm_fences(llvm_ir: str, target_arch: str = "x86_64") -> FenceCheckResult:
    """Analyze LLVM IR for fence correctness."""
    lines = llvm_ir.splitlines()
    fences: List[FenceLocation] = []
    result = FenceCheckResult(
        source_file="<stdin>", compiler=CompilerType.LLVM,
        target_arch=target_arch,
    )

    pat_fence = re.compile(
        r'fence\s+(acquire|release|acq_rel|seq_cst|syncscope\("[^"]*"\)\s+\w+)')
    pat_atomicrmw = re.compile(
        r'atomicrmw\s+(?:xchg|add|sub|and|nand|or|xor|max|min|umax|umin)\s+'
        r'.*\s+(monotonic|acquire|release|acq_rel|seq_cst)')
    pat_cmpxchg = re.compile(
        r'cmpxchg\s+.*\s+(monotonic|acquire|release|acq_rel|seq_cst)\s+'
        r'(monotonic|acquire|release|acq_rel|seq_cst)')
    pat_load_atomic = re.compile(
        r'load\s+atomic\s+.*\s+(monotonic|acquire|seq_cst)')
    pat_store_atomic = re.compile(
        r'store\s+atomic\s+.*\s+(monotonic|release|seq_cst)')

    current_fn: Optional[str] = None
    for idx, line in enumerate(lines):
        lineno = idx + 1
        fn_m = re.match(r'define\s+.*@(\w+)\s*\(', line)
        if fn_m:
            current_fn = fn_m.group(1)

        m = pat_fence.search(line)
        if m:
            ordering_str = m.group(1).strip()
            if ordering_str.startswith("syncscope"):
                inner = re.search(r'\)\s+(\w+)', ordering_str)
                ordering_str = inner.group(1) if inner else "seq_cst"
            kind = _llvm_ordering_to_fence(ordering_str)
            fences.append(FenceLocation(lineno, 1, kind, line.strip(),
                                        current_fn, False))

        m = pat_atomicrmw.search(line)
        if m:
            kind = _llvm_ordering_to_fence(m.group(1))
            fences.append(FenceLocation(lineno, 1, kind, line.strip(),
                                        current_fn, False))

        m = pat_cmpxchg.search(line)
        if m:
            success_ord = m.group(1)
            kind = _llvm_ordering_to_fence(success_ord)
            fences.append(FenceLocation(lineno, 1, kind, line.strip(),
                                        current_fn, False))

        m = pat_load_atomic.search(line)
        if m:
            kind = _llvm_ordering_to_fence(m.group(1))
            fences.append(FenceLocation(lineno, 1, kind, line.strip(),
                                        current_fn, False))

        m = pat_store_atomic.search(line)
        if m:
            kind = _llvm_ordering_to_fence(m.group(1))
            fences.append(FenceLocation(lineno, 1, kind, line.strip(),
                                        current_fn, False))

    result.fences_found = fences

    pairs = _detect_access_pairs(llvm_ir)
    missing: List[MissingBarrier] = []
    for l1, op1, l2, op2, fn in pairs:
        req = _required_ordering_for_pair(op1, op2, target_arch)
        if req == "relaxed":
            continue
        ordering = _extract_ordering_from_line(lines[l1 - 1]) if l1 <= len(lines) else None
        if ordering and _ordering_sufficient(ordering, req):
            continue
        if _has_fence_between(llvm_ir, l1 - 1, l2 - 1):
            continue
        req_kind = _llvm_ordering_to_fence(req)
        missing.append(MissingBarrier(
            line=l1, between_ops=(op1, op2), required_fence=req_kind,
            reason=f"{op1}→{op2} needs at least {req} on {target_arch}",
            severity=4 if req == "seq_cst" else 3,
            fix_suggestion=f"Add 'fence {req}' between lines {l1} and {l2}",
        ))
    result.missing = missing

    redundant: List[RedundantFence] = []
    prev_fence_line: Optional[int] = None
    prev_fence_kind: Optional[FenceKind] = None
    for f in fences:
        if prev_fence_line is not None and f.line - prev_fence_line <= 2:
            if f.fence_kind == prev_fence_kind:
                redundant.append(RedundantFence(
                    line=f.line, fence_kind=f.fence_kind,
                    reason="Consecutive identical fence",
                    can_remove=True,
                    savings_cycles=_fence_cost(f.fence_kind, target_arch),
                ))
        if _is_tso(target_arch) and f.fence_kind in (
                FenceKind.ACQUIRE_BARRIER, FenceKind.RELEASE_BARRIER,
                FenceKind.LOAD_BARRIER):
            redundant.append(RedundantFence(
                line=f.line, fence_kind=f.fence_kind,
                reason=f"Hardware-enforced on TSO ({target_arch})",
                can_remove=True,
                savings_cycles=_fence_cost(f.fence_kind, target_arch),
            ))
        prev_fence_line = f.line
        prev_fence_kind = f.fence_kind
    result.redundant = redundant

    result.total_fence_cost = sum(_fence_cost(f.fence_kind, target_arch)
                                  for f in fences)
    removable = sum(r.savings_cycles for r in redundant if r.can_remove)
    result.optimized_fence_cost = max(0.0, result.total_fence_cost - removable)
    n_issues = len(missing) * 3 + len(redundant)
    result.score = max(0.0, 100.0 - n_issues * 5.0)
    return result


# ---------------------------------------------------------------------------
# Public API — Rust
# ---------------------------------------------------------------------------

def check_rustc_fences(rust_source: str, target_arch: str = "x86_64") -> FenceCheckResult:
    """Analyze Rust source for fence correctness."""
    lines = rust_source.splitlines()
    fences: List[FenceLocation] = []
    result = FenceCheckResult(
        source_file="<stdin>", compiler=CompilerType.RUSTC,
        target_arch=target_arch,
    )

    pat_atomic_new = re.compile(r'Atomic(?:Usize|I32|U32|I64|U64|Bool|Ptr)::new\s*\(')
    pat_load = re.compile(r'\.load\s*\(\s*Ordering::(\w+)\s*\)')
    pat_store = re.compile(r'\.store\s*\([^,]+,\s*Ordering::(\w+)\s*\)')
    pat_cmpxchg = re.compile(
        r'\.compare_exchange(?:_weak)?\s*\([^,]+,[^,]+,\s*Ordering::(\w+)\s*,\s*Ordering::(\w+)\s*\)')
    pat_fetch = re.compile(
        r'\.fetch_(?:add|sub|and|or|xor|nand|min|max|update)\s*\([^,]+,\s*Ordering::(\w+)\s*\)')
    pat_fence = re.compile(r'(?:atomic::)?fence\s*\(\s*Ordering::(\w+)\s*\)')
    pat_cfence = re.compile(r'(?:atomic::)?compiler_fence\s*\(\s*Ordering::(\w+)\s*\)')
    pat_swap = re.compile(r'\.swap\s*\([^,]+,\s*Ordering::(\w+)\s*\)')

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lineno = idx + 1
        fn_name = _current_function(rust_source, idx)
        loop = _in_loop(rust_source, idx)

        m = pat_load.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'\.load')
            fences.append(FenceLocation(lineno, col, kind, stripped, fn_name, loop))

        m = pat_store.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'\.store')
            fences.append(FenceLocation(lineno, col, kind, stripped, fn_name, loop))

        m = pat_cmpxchg.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'\.compare_exchange')
            fences.append(FenceLocation(lineno, col, kind, stripped, fn_name, loop))

        m = pat_fetch.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'\.fetch_')
            fences.append(FenceLocation(lineno, col, kind, stripped, fn_name, loop))

        m = pat_swap.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'\.swap')
            fences.append(FenceLocation(lineno, col, kind, stripped, fn_name, loop))

        m = pat_fence.search(line)
        if m:
            kind = _rust_ordering_to_fence(m.group(1))
            col = _extract_column(line, r'fence\s*\(')
            fences.append(FenceLocation(lineno, col, FenceKind.ATOMIC_FENCE,
                                        stripped, fn_name, loop))

        m = pat_cfence.search(line)
        if m:
            col = _extract_column(line, r'compiler_fence')
            fences.append(FenceLocation(lineno, col, FenceKind.COMPILER_BARRIER,
                                        stripped, fn_name, loop))

    result.fences_found = fences

    # Detect message-passing with Relaxed ordering
    missing: List[MissingBarrier] = []
    thread_fns: Set[str] = set()
    for idx, line in enumerate(lines):
        if _THREAD_FUNC_PATTERNS.search(line):
            fn = _current_function(rust_source, idx)
            if fn:
                thread_fns.add(fn)

    for idx, line in enumerate(lines):
        lineno = idx + 1
        fn = _current_function(rust_source, idx)
        if fn and fn in thread_fns:
            m_store = pat_store.search(line)
            if m_store and m_store.group(1) == "Relaxed":
                for j in range(max(0, idx - 5), min(len(lines), idx + 6)):
                    if j == idx:
                        continue
                    m_load = pat_load.search(lines[j])
                    if m_load and m_load.group(1) == "Relaxed":
                        missing.append(MissingBarrier(
                            line=lineno,
                            between_ops=("store(Relaxed)", "load(Relaxed)"),
                            required_fence=FenceKind.RELEASE_BARRIER,
                            reason="Message-passing pattern needs Release/Acquire",
                            severity=4,
                            fix_suggestion=(
                                f"Use Ordering::Release for store at L{lineno} "
                                f"and Ordering::Acquire for load at L{j + 1}"),
                        ))
                        break

    pairs = _detect_access_pairs(rust_source)
    for l1, op1, l2, op2, fn_ctx in pairs:
        req = _required_ordering_for_pair(op1, op2, target_arch)
        if req == "relaxed":
            continue
        ordering = _extract_ordering_from_line(lines[l1 - 1]) if l1 <= len(lines) else None
        if ordering and _ordering_sufficient(ordering, req):
            continue
        if _has_fence_between(rust_source, l1 - 1, l2 - 1):
            continue
        req_kind = _rust_ordering_to_fence(req)
        missing.append(MissingBarrier(
            line=l1, between_ops=(op1, op2), required_fence=req_kind,
            reason=f"{op1}→{op2} pair needs at least {req} on {target_arch}",
            severity=3,
            fix_suggestion=f"Upgrade ordering to at least {req} on L{l1}",
        ))
    result.missing = missing

    result.redundant = _rust_redundant(fences, target_arch)
    result.total_fence_cost = sum(_fence_cost(f.fence_kind, target_arch)
                                  for f in fences)
    removable = sum(r.savings_cycles for r in result.redundant if r.can_remove)
    result.optimized_fence_cost = max(0.0, result.total_fence_cost - removable)
    n_issues = len(result.missing) * 3 + len(result.redundant)
    result.score = max(0.0, 100.0 - n_issues * 5.0)
    return result


def _rust_redundant(fences: List[FenceLocation], arch: str) -> List[RedundantFence]:
    redundant: List[RedundantFence] = []
    prev: Optional[FenceLocation] = None
    for f in fences:
        if prev and f.line - prev.line <= 2 and f.fence_kind == prev.fence_kind:
            redundant.append(RedundantFence(
                line=f.line, fence_kind=f.fence_kind,
                reason="Back-to-back identical ordering",
                can_remove=True,
                savings_cycles=_fence_cost(f.fence_kind, arch),
            ))
        if _is_tso(arch) and f.fence_kind in (
                FenceKind.ACQUIRE_BARRIER, FenceKind.RELEASE_BARRIER,
                FenceKind.LOAD_BARRIER):
            redundant.append(RedundantFence(
                line=f.line, fence_kind=f.fence_kind,
                reason=f"TSO hardware already enforces this on {arch}",
                can_remove=True,
                savings_cycles=_fence_cost(f.fence_kind, arch),
            ))
        if f.fence_kind == FenceKind.FULL_BARRIER:
            redundant.append(RedundantFence(
                line=f.line, fence_kind=f.fence_kind,
                reason="SeqCst may be stronger than needed; consider AcqRel",
                can_remove=False,
                savings_cycles=_fence_cost(FenceKind.FULL_BARRIER, arch)
                               - _fence_cost(FenceKind.ACQUIRE_BARRIER, arch),
            ))
        prev = f
    return redundant


# ---------------------------------------------------------------------------
# Missing / Redundant analysis (generic C/C++ source)
# ---------------------------------------------------------------------------

def missing_compiler_barriers(source: str, arch: str) -> List[MissingBarrier]:
    """Analyze *source* for access patterns missing barriers on *arch*."""
    missing: List[MissingBarrier] = []
    pairs = _detect_access_pairs(source)

    for l1, op1, l2, op2, fn in pairs:
        req = _required_ordering_for_pair(op1, op2, arch)
        if req == "relaxed":
            continue

        lines = source.splitlines()
        ordering1 = _extract_ordering_from_line(lines[l1 - 1]) if l1 <= len(lines) else None
        if ordering1 and _ordering_sufficient(ordering1, req):
            continue

        if _has_fence_between(source, l1 - 1, l2 - 1):
            continue

        req_kind = _gcc_ordering_to_fence(req)
        severity = 5 if req == "seq_cst" else (4 if req == "acq_rel" else 3)
        missing.append(MissingBarrier(
            line=l1,
            between_ops=(op1, op2),
            required_fence=req_kind,
            reason=f"{op1}→{op2} on {arch} requires at least {req}",
            severity=severity,
            fix_suggestion=(
                f"Insert barrier or upgrade ordering to {req} between "
                f"L{l1} and L{l2}"),
        ))

    # Detect volatile accesses without compiler barrier on weak archs
    if _is_weak(arch):
        vol_pat = re.compile(r'\bvolatile\b')
        lines = source.splitlines()
        prev_volatile_line: Optional[int] = None
        for idx, line in enumerate(lines):
            if vol_pat.search(line):
                if prev_volatile_line is not None and idx - prev_volatile_line < 10:
                    if not _has_fence_between(source, prev_volatile_line, idx):
                        missing.append(MissingBarrier(
                            line=idx + 1,
                            between_ops=("volatile_access", "volatile_access"),
                            required_fence=FenceKind.COMPILER_BARRIER,
                            reason="Consecutive volatile accesses without barrier on weak arch",
                            severity=2,
                            fix_suggestion=f"Add compiler barrier between L{prev_volatile_line + 1} and L{idx + 1}",
                        ))
                prev_volatile_line = idx

    # Shared-variable heuristic: global pointer dereference in thread functions
    lines = source.splitlines()
    shared_pat = re.compile(r'\*\s*(?:shared|global|g_|s_)\w+')
    for idx, line in enumerate(lines):
        fn = _current_function(source, idx)
        if fn and _THREAD_FUNC_PATTERNS.match(fn):
            if shared_pat.search(line):
                ordering = _extract_ordering_from_line(line)
                if not ordering:
                    has_nearby_fence = _has_fence_between(
                        source, max(0, idx - 3), min(len(lines), idx + 3))
                    if not has_nearby_fence:
                        missing.append(MissingBarrier(
                            line=idx + 1,
                            between_ops=("shared_access", "unknown"),
                            required_fence=FenceKind.ACQUIRE_BARRIER,
                            reason="Shared variable access in thread function without ordering",
                            severity=3,
                            fix_suggestion="Use atomic operations or add a fence",
                        ))
    return missing


def redundant_compiler_fences(source: str, arch: str) -> List[RedundantFence]:
    """Find redundant fences in *source* for *arch*."""
    redundant: List[RedundantFence] = []
    lines = source.splitlines()
    fence_lines: List[Tuple[int, FenceKind, str]] = []

    fence_pat = re.compile(
        r'(?:__sync_synchronize|__atomic_thread_fence|asm\s+volatile|'
        r'_mm_mfence|_mm_sfence|_mm_lfence|fence\s*\(|compiler_fence)',
        re.IGNORECASE)
    for idx, line in enumerate(lines):
        if fence_pat.search(line):
            ordering = _extract_ordering_from_line(line)
            if ordering:
                kind = _gcc_ordering_to_fence(ordering)
            elif "__sync_synchronize" in line or "mfence" in line.lower():
                kind = FenceKind.FULL_BARRIER
            elif "sfence" in line.lower():
                kind = FenceKind.STORE_BARRIER
            elif "lfence" in line.lower():
                kind = FenceKind.LOAD_BARRIER
            else:
                kind = FenceKind.COMPILER_BARRIER
            fence_lines.append((idx + 1, kind, line.strip()))

    # Consecutive identical fences
    for i in range(1, len(fence_lines)):
        l_prev, k_prev, _ = fence_lines[i - 1]
        l_cur, k_cur, _ = fence_lines[i]
        if l_cur - l_prev <= 3 and k_cur == k_prev:
            redundant.append(RedundantFence(
                line=l_cur, fence_kind=k_cur,
                reason="Duplicate fence within 3 lines",
                can_remove=True,
                savings_cycles=_fence_cost(k_cur, arch),
            ))

    # SeqCst where AcqRel suffices — heuristic: no total-order requirement
    for lineno, kind, text in fence_lines:
        if kind == FenceKind.FULL_BARRIER:
            ordering = _extract_ordering_from_line(text)
            if ordering == "seq_cst":
                redundant.append(RedundantFence(
                    line=lineno, fence_kind=kind,
                    reason="SeqCst may be reducible to AcqRel",
                    can_remove=False,
                    savings_cycles=(_fence_cost(FenceKind.FULL_BARRIER, arch)
                                    - _fence_cost(FenceKind.ACQUIRE_BARRIER, arch)),
                ))

    # TSO-redundant fences
    if _is_tso(arch):
        for lineno, kind, text in fence_lines:
            if kind in (FenceKind.ACQUIRE_BARRIER, FenceKind.RELEASE_BARRIER,
                        FenceKind.LOAD_BARRIER):
                redundant.append(RedundantFence(
                    line=lineno, fence_kind=kind,
                    reason=f"Hardware-enforced ordering on TSO ({arch})",
                    can_remove=True,
                    savings_cycles=_fence_cost(kind, arch),
                ))

    # AcqRel where only Acquire or Release needed
    for lineno, kind, text in fence_lines:
        if kind == FenceKind.FULL_BARRIER:
            ordering = _extract_ordering_from_line(text)
            if ordering == "acq_rel":
                nearby = lines[max(0, lineno - 4):lineno + 3]
                has_store = any(re.search(r'(?:store|__atomic_store|\.store)', l)
                                for l in nearby)
                has_load = any(re.search(r'(?:load|__atomic_load|\.load)', l)
                               for l in nearby)
                if has_store and not has_load:
                    redundant.append(RedundantFence(
                        line=lineno, fence_kind=kind,
                        reason="AcqRel used but only stores nearby; Release suffices",
                        can_remove=False,
                        savings_cycles=(_fence_cost(FenceKind.FULL_BARRIER, arch)
                                        - _fence_cost(FenceKind.RELEASE_BARRIER, arch)),
                    ))
                elif has_load and not has_store:
                    redundant.append(RedundantFence(
                        line=lineno, fence_kind=kind,
                        reason="AcqRel used but only loads nearby; Acquire suffices",
                        can_remove=False,
                        savings_cycles=(_fence_cost(FenceKind.FULL_BARRIER, arch)
                                        - _fence_cost(FenceKind.ACQUIRE_BARRIER, arch)),
                    ))
    return redundant


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

def optimize_compiler_fences(source: str, arch: str) -> OptimizationResult:
    """Return *source* with minimal correct fences for *arch*."""
    optimized = source
    removed = 0
    added = 0
    replaced = 0
    lines = source.splitlines()

    # Pass 1: remove redundant fences
    redundant = redundant_compiler_fences(source, arch)
    removable_lines: Set[int] = set()
    for r in redundant:
        if r.can_remove:
            removable_lines.add(r.line)
    if removable_lines:
        new_lines: List[str] = []
        for idx, line in enumerate(lines):
            if idx + 1 in removable_lines:
                removed += 1
            else:
                new_lines.append(line)
        optimized = "\n".join(new_lines)
        lines = new_lines

    # Pass 2: downgrade over-strong orderings
    # seq_cst -> acq_rel where safe
    def _replace_ordering(text: str, old: str, new: str) -> Tuple[str, bool]:
        updated = text.replace(old, new, 1)
        return (updated, updated != text)

    new_lines2: List[str] = []
    for line in lines:
        changed = False
        if "__ATOMIC_SEQ_CST" in line and "__atomic_thread_fence" not in line:
            # Keep seq_cst on cmpxchg and standalone fences; downgrade on loads/stores
            if re.search(r'__atomic_(?:load|store|fetch_)', line):
                line, changed = _replace_ordering(line, "__ATOMIC_SEQ_CST", "__ATOMIC_ACQ_REL")
        if "Ordering::SeqCst" in line:
            if re.search(r'\.(?:load|store|fetch_)', line):
                line, changed = _replace_ordering(line, "Ordering::SeqCst", "Ordering::AcqRel")
        if changed:
            replaced += 1
        new_lines2.append(line)
    optimized = "\n".join(new_lines2)

    # Pass 3: further downgrade acq_rel to acquire/release where possible
    new_lines3: List[str] = []
    for line in new_lines2:
        changed = False
        if "__ATOMIC_ACQ_REL" in line:
            if re.search(r'__atomic_load', line):
                line, changed = _replace_ordering(line, "__ATOMIC_ACQ_REL", "__ATOMIC_ACQUIRE")
            elif re.search(r'__atomic_store', line):
                line, changed = _replace_ordering(line, "__ATOMIC_ACQ_REL", "__ATOMIC_RELEASE")
        if "Ordering::AcqRel" in line:
            if re.search(r'\.load\s*\(', line):
                line, changed = _replace_ordering(line, "Ordering::AcqRel", "Ordering::Acquire")
            elif re.search(r'\.store\s*\(', line):
                line, changed = _replace_ordering(line, "Ordering::AcqRel", "Ordering::Release")
        if changed:
            replaced += 1
        new_lines3.append(line)
    optimized = "\n".join(new_lines3)

    # Pass 4: add missing fences
    missing = missing_compiler_barriers(optimized, arch)
    if missing:
        opt_lines = optimized.splitlines()
        insertions: Dict[int, str] = {}
        for mb in missing:
            insert_at = mb.line
            if insert_at not in insertions:
                if "Ordering::" in (opt_lines[insert_at - 1] if insert_at <= len(opt_lines) else ""):
                    insertions[insert_at] = "    atomic::fence(Ordering::Acquire);"
                else:
                    insertions[insert_at] = "    __atomic_thread_fence(__ATOMIC_ACQ_REL);"
                added += 1
        final_lines: List[str] = []
        for idx, line in enumerate(opt_lines):
            if idx + 1 in insertions:
                final_lines.append(insertions[idx + 1])
            final_lines.append(line)
        optimized = "\n".join(final_lines)

    # Remove TSO-redundant fences on x86
    if _is_tso(arch):
        tso_lines: List[str] = []
        for line in optimized.splitlines():
            if re.search(r'(?:_mm_lfence|lfence)', line, re.IGNORECASE):
                removed += 1
                continue
            tso_lines.append(line)
        optimized = "\n".join(tso_lines)

    original_cost = sum(_fence_cost(f.fence_kind, arch)
                        for f in check_gcc_fences(source, arch).fences_found)
    optimized_cost = sum(_fence_cost(f.fence_kind, arch)
                         for f in check_gcc_fences(optimized, arch).fences_found)
    speedup = ((original_cost - optimized_cost) / original_cost * 100.0
               if original_cost > 0 else 0.0)

    return OptimizationResult(
        original_source=source,
        optimized_source=optimized,
        fences_removed=removed,
        fences_added=added,
        fences_replaced=replaced,
        estimated_speedup_pct=max(0.0, speedup),
        correctness_preserved=len(missing_compiler_barriers(optimized, arch)) == 0,
    )


# ---------------------------------------------------------------------------
# Full compilation-unit analysis
# ---------------------------------------------------------------------------

def analyze_compilation_unit(
    source: str,
    compiler: CompilerType,
    arch: str,
    opt_level: OptLevel = OptLevel.O2,
) -> FenceCheckResult:
    """Run the full fence analysis pipeline for a compilation unit."""

    if compiler == CompilerType.GCC:
        result = check_gcc_fences(source, arch)
    elif compiler == CompilerType.LLVM:
        result = check_llvm_fences(source, arch)
    elif compiler == CompilerType.RUSTC:
        result = check_rustc_fences(source, arch)
    elif compiler == CompilerType.MSVC:
        result = _check_msvc_fences(source, arch)
    else:
        result = check_gcc_fences(source, arch)

    result.compiler = compiler

    # Adjust score by optimization level: higher opt → more aggressive reordering
    opt_penalty: Dict[OptLevel, float] = {
        OptLevel.O0: 0.0,
        OptLevel.O1: 2.0,
        OptLevel.O2: 5.0,
        OptLevel.O3: 10.0,
        OptLevel.Os: 4.0,
        OptLevel.Oz: 3.0,
    }
    extra_missing: List[MissingBarrier] = []
    if opt_level in (OptLevel.O2, OptLevel.O3):
        lines = source.splitlines()
        vol_write = re.compile(r'\bvolatile\b.*=')
        for idx, line in enumerate(lines):
            if vol_write.search(line) and not _has_fence_between(
                    source, max(0, idx - 2), min(len(lines), idx + 2)):
                fn = _current_function(source, idx)
                if fn and _THREAD_FUNC_PATTERNS.match(fn):
                    extra_missing.append(MissingBarrier(
                        line=idx + 1,
                        between_ops=("volatile_write", "following_access"),
                        required_fence=FenceKind.COMPILER_BARRIER,
                        reason=(f"At -{opt_level}, compiler may reorder across volatile "
                                "write in thread function"),
                        severity=2,
                        fix_suggestion="Add compiler barrier after volatile write",
                    ))
    result.missing.extend(extra_missing)

    penalty = opt_penalty.get(opt_level, 0.0)
    result.score = max(0.0, result.score - penalty * len(extra_missing))
    return result


def _check_msvc_fences(source: str, arch: str) -> FenceCheckResult:
    """Analyze source with MSVC-specific intrinsics."""
    lines = source.splitlines()
    fences: List[FenceLocation] = []
    result = FenceCheckResult(
        source_file="<stdin>", compiler=CompilerType.MSVC,
        target_arch=arch,
    )

    pat_barrier = re.compile(r'MemoryBarrier\s*\(\s*\)')
    pat_rw_barrier = re.compile(r'_ReadWriteBarrier\s*\(\s*\)')
    pat_read_barrier = re.compile(r'_ReadBarrier\s*\(\s*\)')
    pat_write_barrier = re.compile(r'_WriteBarrier\s*\(\s*\)')
    pat_interlocked = re.compile(
        r'Interlocked(?:Exchange|CompareExchange|Increment|Decrement|'
        r'Add|And|Or|Xor)\w*\s*\(')
    pat_volatile = re.compile(r'_Interlocked\w+')

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        lineno = idx + 1
        fn_name = _current_function(source, idx)
        loop = _in_loop(source, idx)

        if pat_barrier.search(line):
            col = _extract_column(line, r'MemoryBarrier')
            fences.append(FenceLocation(lineno, col, FenceKind.FULL_BARRIER,
                                        stripped, fn_name, loop))
        if pat_rw_barrier.search(line):
            col = _extract_column(line, r'_ReadWriteBarrier')
            fences.append(FenceLocation(lineno, col, FenceKind.COMPILER_BARRIER,
                                        stripped, fn_name, loop))
        if pat_read_barrier.search(line):
            col = _extract_column(line, r'_ReadBarrier')
            fences.append(FenceLocation(lineno, col, FenceKind.LOAD_BARRIER,
                                        stripped, fn_name, loop))
        if pat_write_barrier.search(line):
            col = _extract_column(line, r'_WriteBarrier')
            fences.append(FenceLocation(lineno, col, FenceKind.STORE_BARRIER,
                                        stripped, fn_name, loop))
        if pat_interlocked.search(line):
            col = _extract_column(line, r'Interlocked')
            fences.append(FenceLocation(lineno, col, FenceKind.FULL_BARRIER,
                                        stripped, fn_name, loop))
        if pat_volatile.search(line) and not pat_interlocked.search(line):
            col = _extract_column(line, r'_Interlocked')
            fences.append(FenceLocation(lineno, col, FenceKind.ATOMIC_FENCE,
                                        stripped, fn_name, loop))

    result.fences_found = fences
    result.missing = missing_compiler_barriers(source, arch)
    result.redundant = redundant_compiler_fences(source, arch)
    result.total_fence_cost = sum(_fence_cost(f.fence_kind, arch) for f in fences)
    removable = sum(r.savings_cycles for r in result.redundant if r.can_remove)
    result.optimized_fence_cost = max(0.0, result.total_fence_cost - removable)
    n_issues = len(result.missing) * 3 + len(result.redundant)
    result.score = max(0.0, 100.0 - n_issues * 5.0)
    return result
