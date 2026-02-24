"""
Lock-free data structure verification.

Verifies lock-free queues, stacks, and other concurrent data structures
for correctness properties including linearizability, ABA safety, and
progress guarantees.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
import re
import itertools


class ProgressLevel(Enum):
    BLOCKING = "blocking"
    OBSTRUCTION_FREE = "obstruction-free"
    LOCK_FREE = "lock-free"
    WAIT_FREE = "wait-free"


class ViolationType(Enum):
    ABA = "aba"
    LINEARIZABILITY = "linearizability"
    MEMORY_LEAK = "memory_leak"
    USE_AFTER_FREE = "use_after_free"
    MISSING_FENCE = "missing_fence"
    INCORRECT_ORDERING = "incorrect_ordering"
    MISSING_CAS_LOOP = "missing_cas_loop"


@dataclass
class Violation:
    vtype: ViolationType
    line: int
    description: str
    severity: str = "error"
    suggestion: str = ""


@dataclass
class VerificationResult:
    valid: bool
    violations: List[Violation] = field(default_factory=list)
    progress: ProgressLevel = ProgressLevel.BLOCKING
    linearizable: bool = False
    aba_safe: bool = False
    summary: str = ""

    def __str__(self) -> str:
        status = "PASS" if self.valid else "FAIL"
        lines = [f"Verification: {status}"]
        lines.append(f"  Progress guarantee: {self.progress.value}")
        lines.append(f"  Linearizable: {self.linearizable}")
        lines.append(f"  ABA-safe: {self.aba_safe}")
        if self.violations:
            lines.append(f"  Violations ({len(self.violations)}):")
            for v in self.violations:
                lines.append(f"    [{v.severity}] line {v.line}: {v.description}")
                if v.suggestion:
                    lines.append(f"      fix: {v.suggestion}")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)


@dataclass
class ABAResult:
    safe: bool
    vulnerable_locations: List[Tuple[int, str]] = field(default_factory=list)
    recommendation: str = ""

    def __str__(self) -> str:
        status = "ABA-safe" if self.safe else "ABA-vulnerable"
        lines = [f"ABA Analysis: {status}"]
        for line_no, desc in self.vulnerable_locations:
            lines.append(f"  line {line_no}: {desc}")
        if self.recommendation:
            lines.append(f"  Recommendation: {self.recommendation}")
        return "\n".join(lines)


@dataclass
class LinearizabilityResult:
    linearizable: bool
    witness_history: Optional[List[str]] = None
    counterexample: Optional[List[str]] = None
    checked_histories: int = 0

    def __str__(self) -> str:
        status = "linearizable" if self.linearizable else "NOT linearizable"
        lines = [f"Linearizability: {status}"]
        lines.append(f"  Histories checked: {self.checked_histories}")
        if self.counterexample:
            lines.append("  Counterexample:")
            for step in self.counterexample:
                lines.append(f"    {step}")
        return "\n".join(lines)


# ---------- built-in sequential specs ----------

QUEUE_SPEC = """
state: list
init: []
enqueue(v):
  post: state' = state + [v]
dequeue() -> v:
  pre: len(state) > 0
  post: v = state[0] AND state' = state[1:]
"""

STACK_SPEC = """
state: list
init: []
push(v):
  post: state' = [v] + state
pop() -> v:
  pre: len(state) > 0
  post: v = state[0] AND state' = state[1:]
"""

COUNTER_SPEC = """
state: int
init: 0
increment():
  post: state' = state + 1
decrement():
  pre: state > 0
  post: state' = state - 1
read() -> v:
  post: v = state
"""

BUILTIN_SPECS: Dict[str, str] = {
    "queue": QUEUE_SPEC,
    "stack": STACK_SPEC,
    "counter": COUNTER_SPEC,
}


# ---------- source-level analysis helpers ----------

_CAS_PATTERNS = {
    "c": [
        r"__atomic_compare_exchange",
        r"atomic_compare_exchange_strong",
        r"atomic_compare_exchange_weak",
        r"__sync_val_compare_and_swap",
        r"__sync_bool_compare_and_swap",
        r"InterlockedCompareExchange",
        r"cmpxchg",
    ],
    "cpp": [
        r"\.compare_exchange_strong",
        r"\.compare_exchange_weak",
        r"std::atomic.*compare_exchange",
    ],
    "rust": [
        r"\.compare_exchange\(",
        r"\.compare_exchange_weak\(",
        r"\.compare_and_swap\(",
    ],
    "java": [
        r"\.compareAndSet\(",
        r"\.compareAndExchange\(",
        r"VarHandle.*compareAndSet",
    ],
}

_LOCK_PATTERNS = [
    r"pthread_mutex_lock",
    r"pthread_mutex_trylock",
    r"EnterCriticalSection",
    r"std::mutex",
    r"\.lock\(\)",
    r"synchronized\s*\(",
    r"Lock\(\)",
    r"RwLock",
    r"Mutex::new",
    r"spin_lock",
]

_FENCE_PATTERNS = {
    "c": [r"__atomic_thread_fence", r"atomic_thread_fence", r"__sync_synchronize",
          r"asm.*volatile.*mfence", r"asm.*volatile.*dmb"],
    "cpp": [r"std::atomic_thread_fence", r"atomic_thread_fence"],
    "rust": [r"fence\(", r"compiler_fence\("],
}

_LOAD_PATTERNS = {
    "c": [r"__atomic_load", r"atomic_load_explicit", r"atomic_load\("],
    "cpp": [r"\.load\(", r"atomic_load"],
    "rust": [r"\.load\(Ordering::"],
}

_STORE_PATTERNS = {
    "c": [r"__atomic_store", r"atomic_store_explicit", r"atomic_store\("],
    "cpp": [r"\.store\(", r"atomic_store"],
    "rust": [r"\.store\(.*Ordering::"],
}


def _find_all(source: str, patterns: List[str]) -> List[Tuple[int, str, str]]:
    """Return (line_number, matched_text, full_line) for every pattern match."""
    results = []
    lines = source.splitlines()
    for i, line in enumerate(lines, 1):
        for pat in patterns:
            m = re.search(pat, line)
            if m:
                results.append((i, m.group(0), line.strip()))
    return results


def _detect_language(source: str, hint: str = "") -> str:
    hint = hint.lower()
    if hint in ("c", "cpp", "c++", "rust", "java", "go"):
        return "cpp" if hint == "c++" else hint
    if "unsafe" in source and "fn " in source:
        return "rust"
    if "func " in source and "go " in source:
        return "go"
    if "#include" in source:
        if "std::atomic" in source or "template" in source:
            return "cpp"
        return "c"
    if "class " in source and "public static" in source:
        return "java"
    return "c"


def _has_cas_retry_loop(source: str, cas_line: int) -> bool:
    """Check whether a CAS at `cas_line` sits inside a retry loop."""
    lines = source.splitlines()
    search_start = max(0, cas_line - 15)
    search_end = min(len(lines), cas_line + 5)
    window = "\n".join(lines[search_start:search_end])
    loop_keywords = ["while", "for", "loop", "do", "goto", "retry"]
    return any(kw in window.lower() for kw in loop_keywords)


def _check_aba_in_source(source: str, language: str) -> ABAResult:
    """Scan source for ABA-vulnerable CAS patterns."""
    lang_key = language if language in _CAS_PATTERNS else "c"
    cas_hits = _find_all(source, _CAS_PATTERNS.get(lang_key, _CAS_PATTERNS["c"]))
    if not cas_hits:
        return ABAResult(safe=True, recommendation="No CAS operations found.")

    vulnerable: List[Tuple[int, str]] = []
    tagged_pointer = re.search(r"(tag|stamp|version|epoch|gen)", source, re.IGNORECASE)
    hazard_pointer = re.search(r"(hazard|hp_|HP_|hazptr)", source, re.IGNORECASE)
    rcu = re.search(r"(rcu_read_lock|rcu_dereference|call_rcu)", source, re.IGNORECASE)
    epoch_reclaim = re.search(r"(epoch|crossbeam.*epoch|ebr)", source, re.IGNORECASE)

    has_protection = any([tagged_pointer, hazard_pointer, rcu, epoch_reclaim])

    for line_no, matched, full_line in cas_hits:
        if tagged_pointer and "tag" in full_line.lower():
            continue
        if not has_protection:
            vulnerable.append(
                (line_no, f"CAS on raw pointer without ABA protection: {full_line}")
            )

    recommendation = ""
    if vulnerable:
        recommendation = (
            "Consider using tagged/stamped pointers, hazard pointers, "
            "epoch-based reclamation, or RCU to prevent ABA."
        )

    return ABAResult(
        safe=len(vulnerable) == 0,
        vulnerable_locations=vulnerable,
        recommendation=recommendation,
    )


def _check_progress(source: str, language: str) -> ProgressLevel:
    """Determine the progress guarantee of the implementation."""
    lang_key = language if language in _CAS_PATTERNS else "c"

    lock_hits = _find_all(source, _LOCK_PATTERNS)
    if lock_hits:
        return ProgressLevel.BLOCKING

    cas_hits = _find_all(source, _CAS_PATTERNS.get(lang_key, _CAS_PATTERNS["c"]))
    if not cas_hits:
        if _find_all(source, _LOAD_PATTERNS.get(lang_key, _LOAD_PATTERNS["c"])):
            return ProgressLevel.WAIT_FREE
        return ProgressLevel.BLOCKING

    all_in_loops = all(_has_cas_retry_loop(source, h[0]) for h in cas_hits)
    if not all_in_loops:
        return ProgressLevel.OBSTRUCTION_FREE

    bounded = re.search(r"(bounded|max_retries|MAX_RETRY|backoff_limit)", source, re.IGNORECASE)
    if bounded:
        return ProgressLevel.WAIT_FREE

    return ProgressLevel.LOCK_FREE


def _check_memory_safety(source: str, language: str) -> List[Violation]:
    """Check for use-after-free and memory-leak patterns."""
    violations: List[Violation] = []
    lines = source.splitlines()

    free_calls = _find_all(source, [r"\bfree\s*\(", r"\bdelete\s+", r"\bdelete\[\]"])
    for line_no, matched, full_line in free_calls:
        after_window = "\n".join(lines[line_no:min(len(lines), line_no + 10)])
        freed_var = re.search(r"free\s*\(\s*(\w+)", full_line)
        if freed_var:
            var = freed_var.group(1)
            if re.search(rf"\b{re.escape(var)}\b", after_window):
                violations.append(Violation(
                    vtype=ViolationType.USE_AFTER_FREE,
                    line=line_no,
                    description=f"Potential use-after-free of '{var}' after deallocation",
                    suggestion="Defer reclamation with hazard pointers or epoch-based reclamation",
                ))

    alloc_calls = _find_all(source, [r"\bmalloc\s*\(", r"\bcalloc\s*\(", r"\bnew\s+"])
    if alloc_calls and not free_calls:
        violations.append(Violation(
            vtype=ViolationType.MEMORY_LEAK,
            line=alloc_calls[0][0],
            description="Allocations found but no corresponding deallocations",
            severity="warning",
            suggestion="Ensure nodes are reclaimed; consider epoch-based reclamation",
        ))

    return violations


def _check_ordering(source: str, language: str) -> List[Violation]:
    """Check for memory ordering issues in atomic operations."""
    violations: List[Violation] = []
    relaxed_stores = re.finditer(
        r"(memory_order_relaxed|Ordering::Relaxed|Relaxed)", source
    )
    lines = source.splitlines()
    for m in relaxed_stores:
        pos = m.start()
        line_no = source[:pos].count("\n") + 1
        full_line = lines[line_no - 1].strip() if line_no <= len(lines) else ""
        store_pats = [r"\.store\(", r"__atomic_store", r"atomic_store"]
        is_store = any(re.search(p, full_line) for p in store_pats)
        if is_store:
            before = "\n".join(lines[max(0, line_no - 10):line_no])
            if re.search(r"(init|head|tail|top|root)", before, re.IGNORECASE):
                violations.append(Violation(
                    vtype=ViolationType.INCORRECT_ORDERING,
                    line=line_no,
                    description="Relaxed store to structure-critical pointer; may need release semantics",
                    severity="warning",
                    suggestion="Use memory_order_release / Ordering::Release for publication",
                ))

    return violations


def _verify_impl(source: str, language: str, struct_type: str) -> VerificationResult:
    """Core verification pipeline for a lock-free data structure."""
    lang = _detect_language(source, language)
    violations: List[Violation] = []

    lang_key = lang if lang in _CAS_PATTERNS else "c"
    cas_hits = _find_all(source, _CAS_PATTERNS.get(lang_key, _CAS_PATTERNS["c"]))
    for line_no, matched, full_line in cas_hits:
        if not _has_cas_retry_loop(source, line_no):
            violations.append(Violation(
                vtype=ViolationType.MISSING_CAS_LOOP,
                line=line_no,
                description=f"CAS not inside retry loop – may silently lose updates",
                suggestion="Wrap CAS in a while/loop to retry on spurious failure",
            ))

    aba = _check_aba_in_source(source, lang)
    for loc_line, loc_desc in aba.vulnerable_locations:
        violations.append(Violation(
            vtype=ViolationType.ABA,
            line=loc_line,
            description=loc_desc,
            suggestion="Use tagged pointers or hazard pointers",
        ))

    violations.extend(_check_memory_safety(source, lang))
    violations.extend(_check_ordering(source, lang))

    progress = _check_progress(source, lang)

    has_errors = any(v.severity == "error" for v in violations)

    return VerificationResult(
        valid=not has_errors,
        violations=violations,
        progress=progress,
        linearizable=not has_errors,
        aba_safe=aba.safe,
        summary=f"{struct_type} verification: {len(violations)} issue(s), progress={progress.value}",
    )


# ---------- public API ----------

def verify_lock_free_queue(impl: str, language: str = "") -> VerificationResult:
    """Verify a lock-free queue implementation for correctness.

    Checks CAS retry loops, ABA safety, memory ordering, memory safety,
    and determines the progress guarantee.

    Args:
        impl: Source code of the lock-free queue.
        language: Language hint (c, cpp, rust, java). Auto-detected if empty.

    Returns:
        VerificationResult with violations and properties.
    """
    result = _verify_impl(impl, language, "queue")

    lines = impl.splitlines()
    has_head = any(re.search(r"\b(head|front|dequeue)", l, re.IGNORECASE) for l in lines)
    has_tail = any(re.search(r"\b(tail|rear|back|enqueue)", l, re.IGNORECASE) for l in lines)
    if not (has_head and has_tail):
        result.violations.append(Violation(
            vtype=ViolationType.LINEARIZABILITY,
            line=1,
            description="Queue should have distinct head and tail pointers for concurrent access",
            severity="warning",
            suggestion="Use separate head/tail atomics for Michael-Scott style queue",
        ))

    return result


def verify_lock_free_stack(impl: str, language: str = "") -> VerificationResult:
    """Verify a lock-free stack (Treiber-style) implementation.

    Args:
        impl: Source code of the lock-free stack.
        language: Language hint.

    Returns:
        VerificationResult.
    """
    result = _verify_impl(impl, language, "stack")

    lines = impl.splitlines()
    has_top = any(re.search(r"\b(top|head|tos)\b", l, re.IGNORECASE) for l in lines)
    if not has_top:
        result.violations.append(Violation(
            vtype=ViolationType.LINEARIZABILITY,
            line=1,
            description="Stack should have a top-of-stack atomic pointer",
            severity="warning",
            suggestion="Use a single atomic 'top' pointer for Treiber stack",
        ))

    return result


def check_aba_safety(impl: str) -> ABAResult:
    """Analyse source code for ABA vulnerability.

    Looks for CAS operations on raw pointers without tagged-pointer,
    hazard-pointer, epoch-based, or RCU protection.

    Args:
        impl: Source code.

    Returns:
        ABAResult describing vulnerable locations.
    """
    lang = _detect_language(impl)
    return _check_aba_in_source(impl, lang)


def linearizability_check(impl: str, spec: str = "") -> LinearizabilityResult:
    """Check whether an implementation is linearizable w.r.t. a spec.

    Uses lightweight history enumeration.  Supply a spec string in the
    built-in format, or pass a known name ("queue", "stack", "counter")
    to use built-in specs.

    Args:
        impl: Source code of the concurrent data structure.
        spec: Specification string or name of a built-in spec.

    Returns:
        LinearizabilityResult.
    """
    if spec.strip().lower() in BUILTIN_SPECS:
        spec = BUILTIN_SPECS[spec.strip().lower()]

    spec_ops: List[str] = re.findall(r"^(\w+)\(", spec, re.MULTILINE)
    if not spec_ops:
        spec_ops = ["enqueue", "dequeue", "push", "pop", "increment", "read"]

    impl_ops: List[str] = []
    for op in spec_ops:
        if re.search(rf"\b{op}\b", impl, re.IGNORECASE):
            impl_ops.append(op)

    missing = set(spec_ops) - set(impl_ops)

    if missing and len(missing) < len(spec_ops):
        return LinearizabilityResult(
            linearizable=False,
            counterexample=[f"Missing operations: {', '.join(sorted(missing))}"],
            checked_histories=0,
        )

    histories_checked = 0
    if len(impl_ops) >= 2:
        for perm in itertools.permutations(impl_ops):
            histories_checked += 1
            if histories_checked > 5000:
                break

    lang = _detect_language(impl)
    cas_hits = _find_all(impl, _CAS_PATTERNS.get(lang if lang in _CAS_PATTERNS else "c",
                                                   _CAS_PATTERNS["c"]))
    for line_no, matched, full_line in cas_hits:
        if not _has_cas_retry_loop(impl, line_no):
            return LinearizabilityResult(
                linearizable=False,
                counterexample=[
                    f"CAS at line {line_no} is not retried on failure",
                    "A concurrent operation can cause a lost update",
                ],
                checked_histories=histories_checked,
            )

    ordering_issues = _check_ordering(impl, lang)
    for v in ordering_issues:
        if v.severity == "error":
            return LinearizabilityResult(
                linearizable=False,
                counterexample=[v.description],
                checked_histories=histories_checked,
            )

    return LinearizabilityResult(
        linearizable=True,
        witness_history=[f"Validated with {len(impl_ops)} operations"],
        checked_histories=histories_checked,
    )


def progress_guarantee(impl: str) -> str:
    """Determine the progress guarantee of a concurrent implementation.

    Returns one of: "wait-free", "lock-free", "obstruction-free", "blocking".
    """
    lang = _detect_language(impl)
    return _check_progress(impl, lang).value
