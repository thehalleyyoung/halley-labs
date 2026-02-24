"""
Concurrency analyzer for detecting deadlocks, livelocks, priority inversion,
starvation, and other concurrency hazards in multi-threaded source code.

Performs static analysis across C, C++, Rust, Go, Java, and Python to produce
a comprehensive concurrency report with a thread-safety score.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
import re


class Severity(Enum):
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class Language(Enum):
    C = "c"
    CPP = "cpp"
    RUST = "rust"
    GO = "go"
    JAVA = "java"
    PYTHON = "python"


class HazardType(Enum):
    DEADLOCK = "deadlock"
    LIVELOCK = "livelock"
    PRIORITY_INVERSION = "priority_inversion"
    STARVATION = "starvation"
    DATA_RACE = "data_race"
    ATOMICITY_VIOLATION = "atomicity_violation"
    ORDER_VIOLATION = "order_violation"


@dataclass
class LockAcquisition:
    lock_name: str
    line: int
    function: str = ""
    thread_context: str = ""

    def __str__(self) -> str:
        loc = f" in {self.function}" if self.function else ""
        return f"lock '{self.lock_name}' at line {self.line}{loc}"


@dataclass
class Deadlock:
    cycle: List[LockAcquisition]
    severity: Severity = Severity.CRITICAL
    description: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        chain = " -> ".join(str(a) for a in self.cycle)
        return f"Deadlock: {chain} | {self.description}"


@dataclass
class Livelock:
    involved_threads: List[str]
    retry_pattern: str = ""
    line: int = 0
    severity: Severity = Severity.HIGH
    description: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        threads = ", ".join(self.involved_threads)
        return f"Livelock at line {self.line}: threads [{threads}] — {self.description}"


@dataclass
class PriorityInversion:
    high_priority: str
    low_priority: str
    shared_resource: str
    line: int = 0
    severity: Severity = Severity.HIGH
    description: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        return (f"Priority inversion at line {self.line}: "
                f"high={self.high_priority}, low={self.low_priority}, "
                f"resource={self.shared_resource}")


@dataclass
class Starvation:
    starved_thread: str
    contended_resource: str
    line: int = 0
    severity: Severity = Severity.MEDIUM
    description: str = ""
    suggestion: str = ""

    def __str__(self) -> str:
        return (f"Starvation at line {self.line}: thread '{self.starved_thread}' "
                f"on resource '{self.contended_resource}'")


@dataclass
class SyncPrimitive:
    name: str
    kind: str  # mutex, rwlock, semaphore, condvar, channel, atomic
    line: int = 0
    is_properly_released: bool = True
    scope: str = ""

    def __str__(self) -> str:
        status = "OK" if self.is_properly_released else "LEAK"
        return f"{self.kind} '{self.name}' at line {self.line} [{status}]"


@dataclass
class SyncAudit:
    primitives: List[SyncPrimitive] = field(default_factory=list)
    unreleased_locks: List[SyncPrimitive] = field(default_factory=list)
    redundant_locks: List[SyncPrimitive] = field(default_factory=list)
    missing_barriers: List[int] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Sync audit: {len(self.primitives)} primitives found"]
        if self.unreleased_locks:
            lines.append(f"  {len(self.unreleased_locks)} unreleased lock(s)")
        if self.redundant_locks:
            lines.append(f"  {len(self.redundant_locks)} redundant lock(s)")
        if self.recommendations:
            lines.append(f"  {len(self.recommendations)} recommendation(s)")
        return "\n".join(lines)


@dataclass
class ConcurrencyReport:
    language: Language
    deadlocks: List[Deadlock] = field(default_factory=list)
    livelocks: List[Livelock] = field(default_factory=list)
    priority_inversions: List[PriorityInversion] = field(default_factory=list)
    starvation_risks: List[Starvation] = field(default_factory=list)
    sync_audit: Optional[SyncAudit] = None
    thread_safety_score: float = 100.0
    summary: str = ""

    def __str__(self) -> str:
        lines = [f"Concurrency Report ({self.language.value}) — score: {self.thread_safety_score:.1f}/100"]
        lines.append(f"  Deadlocks: {len(self.deadlocks)}")
        lines.append(f"  Livelocks: {len(self.livelocks)}")
        lines.append(f"  Priority inversions: {len(self.priority_inversions)}")
        lines.append(f"  Starvation risks: {len(self.starvation_risks)}")
        if self.summary:
            lines.append(f"  Summary: {self.summary}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Language-specific regex patterns
# ---------------------------------------------------------------------------

_LOCK_PATTERNS: Dict[str, List[Tuple[str, str]]] = {
    "c": [
        (r'pthread_mutex_lock\s*\(\s*&?\s*(\w+)', "mutex"),
        (r'pthread_rwlock_wrlock\s*\(\s*&?\s*(\w+)', "rwlock"),
        (r'pthread_rwlock_rdlock\s*\(\s*&?\s*(\w+)', "rwlock"),
        (r'pthread_spin_lock\s*\(\s*&?\s*(\w+)', "spinlock"),
        (r'sem_wait\s*\(\s*&?\s*(\w+)', "semaphore"),
    ],
    "cpp": [
        (r'(\w+)\.lock\s*\(\s*\)', "mutex"),
        (r'std::lock_guard<[^>]*>\s+(\w+)', "lock_guard"),
        (r'std::unique_lock<[^>]*>\s+(\w+)', "unique_lock"),
        (r'std::scoped_lock[^(]*\(([^)]+)\)', "scoped_lock"),
        (r'std::shared_lock<[^>]*>\s+(\w+)', "shared_lock"),
    ],
    "rust": [
        (r'(\w+)\.lock\(\)', "mutex"),
        (r'(\w+)\.read\(\)', "rwlock"),
        (r'(\w+)\.write\(\)', "rwlock"),
    ],
    "go": [
        (r'(\w+)\.Lock\(\)', "mutex"),
        (r'(\w+)\.RLock\(\)', "rwmutex"),
    ],
    "java": [
        (r'(\w+)\.lock\(\)', "reentrant_lock"),
        (r'synchronized\s*\(\s*(\w+)\s*\)', "synchronized"),
        (r'(\w+)\.acquire\(\)', "semaphore"),
    ],
    "python": [
        (r'(\w+)\.acquire\(\)', "lock"),
        (r'with\s+(\w+)\s*:', "context_lock"),
    ],
}

_UNLOCK_PATTERNS: Dict[str, List[str]] = {
    "c": [
        r'pthread_mutex_unlock\s*\(\s*&?\s*(\w+)',
        r'pthread_rwlock_unlock\s*\(\s*&?\s*(\w+)',
        r'pthread_spin_unlock\s*\(\s*&?\s*(\w+)',
        r'sem_post\s*\(\s*&?\s*(\w+)',
    ],
    "cpp": [
        r'(\w+)\.unlock\s*\(\s*\)',
    ],
    "rust": [],  # RAII-based; drop handles unlock
    "go": [
        r'(\w+)\.Unlock\(\)',
        r'(\w+)\.RUnlock\(\)',
    ],
    "java": [
        r'(\w+)\.unlock\(\)',
        r'(\w+)\.release\(\)',
    ],
    "python": [
        r'(\w+)\.release\(\)',
    ],
}

_THREAD_CREATE_PATTERNS: Dict[str, List[str]] = {
    "c": [r'pthread_create\s*\(', r'thrd_create\s*\('],
    "cpp": [r'std::thread\s+(\w+)', r'std::async\s*\(', r'std::jthread\s+(\w+)'],
    "rust": [r'thread::spawn\s*\(', r'rayon::', r'tokio::spawn\s*\('],
    "go": [r'go\s+\w+\s*\(', r'go\s+func\s*\('],
    "java": [r'new\s+Thread\s*\(', r'\.start\s*\(\)', r'ExecutorService',
             r'\.submit\s*\(', r'CompletableFuture'],
    "python": [r'threading\.Thread\s*\(', r'Thread\s*\(target=',
               r'ProcessPoolExecutor', r'ThreadPoolExecutor',
               r'asyncio\.create_task'],
}

_RETRY_PATTERNS = [
    r'while\s*\(\s*!.*try_lock',
    r'while\s*\(\s*!.*compare_exchange',
    r'while\s*\(\s*!.*CAS\b',
    r'while\s*\(.*\.compareAndSet\s*\(',
    r'for\s*\(.*;.*retry',
    r'while\s+True.*acquire',
]

_PRIORITY_PATTERNS: Dict[str, List[str]] = {
    "c": [r'pthread_setschedparam\s*\(', r'sched_setscheduler\s*\('],
    "cpp": [r'SetThreadPriority\s*\(', r'pthread_setschedparam\s*\('],
    "java": [r'\.setPriority\s*\(\s*(\d+)\s*\)', r'Thread\.(MAX|MIN|NORM)_PRIORITY'],
    "python": [r'os\.nice\s*\(', r'os\.setpriority\s*\('],
    "go": [r'runtime\.LockOSThread'],
    "rust": [r'thread_priority::'],
}


def _resolve_language(lang: str) -> Language:
    """Normalize a language string to Language enum."""
    mapping = {
        "c": Language.C, "cpp": Language.CPP, "c++": Language.CPP,
        "rust": Language.RUST, "go": Language.GO,
        "java": Language.JAVA, "python": Language.PYTHON, "py": Language.PYTHON,
    }
    return mapping.get(lang.lower(), Language.C)


def _extract_functions(source: str) -> List[Tuple[str, int, int]]:
    """Return list of (func_name, start_line, end_line)."""
    results: List[Tuple[str, int, int]] = []
    # Matches typical C/C++/Java/Go/Rust function definitions
    pattern = re.compile(
        r'(?:func|fn|void|int|def|public|private|protected|static)?\s+'
        r'(\w+)\s*\([^)]*\)\s*[{:]',
        re.MULTILINE,
    )
    lines = source.split('\n')
    for m in pattern.finditer(source):
        start = source[:m.start()].count('\n') + 1
        name = m.group(1)
        # Find matching brace end (simplified)
        depth = 0
        end = start
        for i in range(start - 1, len(lines)):
            depth += lines[i].count('{') - lines[i].count('}')
            if depth <= 0 and i > start - 1:
                end = i + 1
                break
        else:
            end = len(lines)
        results.append((name, start, end))
    return results


def _find_lock_acquisitions(source: str, lang: str) -> List[LockAcquisition]:
    """Find all lock acquisition sites in source."""
    key = lang if lang in _LOCK_PATTERNS else "c"
    acquisitions: List[LockAcquisition] = []
    functions = _extract_functions(source)
    lines = source.split('\n')

    for pattern_str, kind in _LOCK_PATTERNS.get(key, []):
        pattern = re.compile(pattern_str)
        for i, line_text in enumerate(lines, 1):
            m = pattern.search(line_text)
            if m:
                lock_name = m.group(1).strip()
                func = ""
                for fname, fstart, fend in functions:
                    if fstart <= i <= fend:
                        func = fname
                        break
                acquisitions.append(LockAcquisition(
                    lock_name=lock_name, line=i, function=func,
                ))
    return acquisitions


def _find_unlock_sites(source: str, lang: str) -> Dict[str, List[int]]:
    """Return dict mapping lock_name -> list of unlock line numbers."""
    key = lang if lang in _UNLOCK_PATTERNS else "c"
    unlocks: Dict[str, List[int]] = {}
    lines = source.split('\n')
    for pattern_str in _UNLOCK_PATTERNS.get(key, []):
        pattern = re.compile(pattern_str)
        for i, line_text in enumerate(lines, 1):
            m = pattern.search(line_text)
            if m:
                name = m.group(1).strip()
                unlocks.setdefault(name, []).append(i)
    return unlocks


def _build_lock_order_graph(acquisitions: List[LockAcquisition]) -> Dict[str, Set[str]]:
    """Build a lock ordering graph: edge A->B means A acquired before B in same function."""
    graph: Dict[str, Set[str]] = {}
    by_func: Dict[str, List[LockAcquisition]] = {}
    for acq in acquisitions:
        key = acq.function or "__global__"
        by_func.setdefault(key, []).append(acq)

    for func_acqs in by_func.values():
        sorted_acqs = sorted(func_acqs, key=lambda a: a.line)
        held: List[str] = []
        for acq in sorted_acqs:
            for h in held:
                graph.setdefault(h, set()).add(acq.lock_name)
            held.append(acq.lock_name)
    return graph


def _find_cycles(graph: Dict[str, Set[str]]) -> List[List[str]]:
    """Find cycles in a directed graph using DFS."""
    cycles: List[List[str]] = []
    visited: Set[str] = set()
    rec_stack: Set[str] = set()
    path: List[str] = []

    def dfs(node: str) -> None:
        visited.add(node)
        rec_stack.add(node)
        path.append(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor)
            elif neighbor in rec_stack:
                idx = path.index(neighbor)
                cycle = path[idx:] + [neighbor]
                cycles.append(cycle)
        path.pop()
        rec_stack.discard(node)

    for node in graph:
        if node not in visited:
            dfs(node)
    return cycles


def detect_deadlocks(source: str, language: str = "c") -> List[Deadlock]:
    """Detect potential deadlocks through lock-ordering analysis."""
    lang = _resolve_language(language).value
    acquisitions = _find_lock_acquisitions(source, lang)
    if not acquisitions:
        return []

    graph = _build_lock_order_graph(acquisitions)
    cycles = _find_cycles(graph)
    deadlocks: List[Deadlock] = []

    acq_map: Dict[str, LockAcquisition] = {}
    for a in acquisitions:
        acq_map.setdefault(a.lock_name, a)

    for cycle in cycles:
        chain = []
        for lock_name in cycle:
            if lock_name in acq_map:
                chain.append(acq_map[lock_name])
            else:
                chain.append(LockAcquisition(lock_name=lock_name, line=0))
        lock_names = [a.lock_name for a in chain]
        deadlocks.append(Deadlock(
            cycle=chain,
            description=f"Circular lock dependency: {' -> '.join(lock_names)}",
            suggestion="Enforce a consistent global lock ordering or use std::scoped_lock / try_lock.",
        ))
    return deadlocks


def detect_livelocks(source: str, language: str = "c") -> List[Livelock]:
    """Detect potential livelocks from retry/spin patterns."""
    livelocks: List[Livelock] = []
    lines = source.split('\n')

    for pattern_str in _RETRY_PATTERNS:
        pattern = re.compile(pattern_str, re.IGNORECASE)
        for i, line_text in enumerate(lines, 1):
            if pattern.search(line_text):
                # Check for backoff
                context = "\n".join(lines[max(0, i - 1):min(len(lines), i + 10)])
                has_backoff = bool(re.search(r'sleep|yield|backoff|delay|nanosleep|usleep', context))
                has_bound = bool(re.search(r'max_retries|MAX_RETRY|retry_count|attempts|limit', context))

                if not has_backoff and not has_bound:
                    livelocks.append(Livelock(
                        involved_threads=["unknown"],
                        retry_pattern=line_text.strip(),
                        line=i,
                        severity=Severity.HIGH,
                        description="Unbounded retry loop without backoff — may livelock under contention",
                        suggestion="Add exponential backoff or a retry limit.",
                    ))
                elif not has_backoff:
                    livelocks.append(Livelock(
                        involved_threads=["unknown"],
                        retry_pattern=line_text.strip(),
                        line=i,
                        severity=Severity.MEDIUM,
                        description="Retry loop without backoff (bounded) — contention risk",
                        suggestion="Consider adding exponential backoff.",
                    ))
    return livelocks


def detect_priority_inversion(source: str, language: str = "c") -> List[PriorityInversion]:
    """Detect potential priority inversion scenarios."""
    lang = _resolve_language(language).value
    inversions: List[PriorityInversion] = []
    lines = source.split('\n')

    # Check whether code uses priority settings alongside shared locks
    priority_lines: List[int] = []
    for pattern_str in _PRIORITY_PATTERNS.get(lang, []):
        pattern = re.compile(pattern_str)
        for i, line_text in enumerate(lines, 1):
            if pattern.search(line_text):
                priority_lines.append(i)

    if not priority_lines:
        return inversions

    acquisitions = _find_lock_acquisitions(source, lang)
    if not acquisitions:
        return inversions

    # Heuristic: if multiple priority levels share the same lock, flag it
    lock_names = {a.lock_name for a in acquisitions}
    for lock_name in lock_names:
        lock_lines = [a.line for a in acquisitions if a.lock_name == lock_name]
        for pline in priority_lines:
            # Priority setting near a lock acquisition
            for ll in lock_lines:
                if abs(pline - ll) < 30:
                    inversions.append(PriorityInversion(
                        high_priority="high-priority thread",
                        low_priority="low-priority thread",
                        shared_resource=lock_name,
                        line=ll,
                        description=(
                            f"Lock '{lock_name}' used near priority configuration — "
                            f"risk of priority inversion"
                        ),
                        suggestion="Use priority inheritance mutexes (PTHREAD_PRIO_INHERIT) or priority ceiling.",
                    ))
                    break
    return inversions


def detect_starvation(source: str, language: str = "c") -> List[Starvation]:
    """Detect potential starvation from unfair scheduling or lock patterns."""
    lang = _resolve_language(language).value
    starvation_risks: List[Starvation] = []
    lines = source.split('\n')

    # Pattern 1: writer-preferred rwlocks starving readers (or vice versa)
    rwlock_pattern_c = re.compile(r'pthread_rwlock_(wr|rd)lock\s*\(\s*&?\s*(\w+)')
    rwlock_uses: Dict[str, Set[str]] = {}
    for i, line_text in enumerate(lines, 1):
        m = rwlock_pattern_c.search(line_text)
        if m:
            kind, name = m.group(1), m.group(2)
            rwlock_uses.setdefault(name, set()).add(kind)

    for name, kinds in rwlock_uses.items():
        if "wr" in kinds and "rd" in kinds:
            starvation_risks.append(Starvation(
                starved_thread="reader or writer threads",
                contended_resource=name,
                description=(
                    f"rwlock '{name}' used for both read and write — "
                    f"default policy may starve one side"
                ),
                suggestion="Set PTHREAD_RWLOCK_PREFER_WRITER_NONRECURSIVE_NP or use fair lock.",
            ))

    # Pattern 2: spin locks held for long durations
    spinlock_pattern = re.compile(r'pthread_spin_lock\s*\(\s*&?\s*(\w+)')
    for i, line_text in enumerate(lines, 1):
        if spinlock_pattern.search(line_text):
            context = "\n".join(lines[i:min(len(lines), i + 20)])
            if re.search(r'(sleep|malloc|printf|write\(|read\(|fwrite|send|recv)', context):
                name = spinlock_pattern.search(line_text).group(1)
                starvation_risks.append(Starvation(
                    starved_thread="waiting threads",
                    contended_resource=name,
                    line=i,
                    severity=Severity.HIGH,
                    description=f"Spinlock '{name}' held during blocking operation",
                    suggestion="Replace spinlock with mutex or move blocking I/O outside critical section.",
                ))

    # Pattern 3: unfair trylock loops
    trylock_pattern = re.compile(r'(try_lock|tryLock|TryLock|pthread_mutex_trylock)')
    for i, line_text in enumerate(lines, 1):
        if trylock_pattern.search(line_text):
            context_before = "\n".join(lines[max(0, i - 5):i])
            if re.search(r'while\s*\(', context_before):
                starvation_risks.append(Starvation(
                    starved_thread="contending threads",
                    contended_resource="unknown",
                    line=i,
                    severity=Severity.MEDIUM,
                    description="Busy-wait trylock loop — threads with lower scheduling priority may starve",
                    suggestion="Use blocking lock or add fairness mechanism.",
                ))

    return starvation_risks


def synchronization_audit(source: str, language: str = "c") -> SyncAudit:
    """Audit synchronization primitive usage for correctness and completeness."""
    lang = _resolve_language(language).value
    audit = SyncAudit()

    acquisitions = _find_lock_acquisitions(source, lang)
    unlocks = _find_unlock_sites(source, lang)
    lines = source.split('\n')

    # Build primitive list
    seen: Set[str] = set()
    for acq in acquisitions:
        if acq.lock_name not in seen:
            seen.add(acq.lock_name)
            has_unlock = acq.lock_name in unlocks
            # RAII languages auto-release
            if lang in ("rust", "cpp"):
                has_unlock = True
            audit.primitives.append(SyncPrimitive(
                name=acq.lock_name,
                kind="mutex",
                line=acq.line,
                is_properly_released=has_unlock,
            ))

    # Detect condition variables
    condvar_patterns = {
        "c": r'pthread_cond_(wait|signal|broadcast)\s*\(\s*&?\s*(\w+)',
        "cpp": r'(\w+)\.(wait|notify_one|notify_all)\s*\(',
        "java": r'(\w+)\.(await|signal|signalAll|wait|notify|notifyAll)\s*\(',
        "python": r'(\w+)\.(wait|notify|notify_all)\s*\(',
        "go": r'(\w+)\.(Wait|Signal|Broadcast)\s*\(',
    }
    cv_pat = condvar_patterns.get(lang, "")
    if cv_pat:
        for i, line_text in enumerate(lines, 1):
            m = re.search(cv_pat, line_text)
            if m:
                name = m.group(1) if lang != "c" else m.group(2)
                if name not in seen:
                    seen.add(name)
                    audit.primitives.append(SyncPrimitive(
                        name=name, kind="condvar", line=i, is_properly_released=True,
                    ))

    # Detect atomics
    atomic_patterns = {
        "c": r'atomic_(load|store|fetch_add|compare_exchange)\w*\s*\(',
        "cpp": r'(std::atomic|\.load|\.store|\.fetch_add|\.compare_exchange)',
        "rust": r'(AtomicUsize|AtomicBool|AtomicPtr|Ordering::)',
        "go": r'atomic\.(Load|Store|Add|CompareAndSwap)',
        "java": r'(AtomicInteger|AtomicLong|AtomicReference|AtomicBoolean)',
        "python": r'# no native atomics',
    }
    atom_pat = atomic_patterns.get(lang, "")
    if atom_pat:
        for i, line_text in enumerate(lines, 1):
            if re.search(atom_pat, line_text):
                if "__atomic__" not in seen:
                    seen.add("__atomic__")
                    audit.primitives.append(SyncPrimitive(
                        name="atomic operations", kind="atomic", line=i,
                    ))
                break

    # Check for unreleased locks
    audit.unreleased_locks = [p for p in audit.primitives
                              if not p.is_properly_released and p.kind != "atomic"]

    # Recommendations
    if audit.unreleased_locks:
        audit.recommendations.append(
            "Use RAII wrappers (lock_guard / scoped_lock) to ensure locks are always released."
        )

    thread_count = 0
    for pat in _THREAD_CREATE_PATTERNS.get(lang, []):
        thread_count += len(re.findall(pat, source))

    if thread_count > 0 and not audit.primitives:
        audit.recommendations.append(
            "Thread creation detected but no synchronization primitives found — "
            "potential unsynchronized shared state."
        )

    if len(audit.primitives) > 5:
        audit.recommendations.append(
            "High number of synchronization primitives — consider simplifying with "
            "higher-level abstractions (channels, actors, STM)."
        )

    return audit


def thread_safety_score(source: str, language: str = "c") -> float:
    """Compute a thread-safety score from 0 (unsafe) to 100 (safe)."""
    lang = _resolve_language(language).value
    score = 100.0

    deadlocks = detect_deadlocks(source, language)
    livelocks = detect_livelocks(source, language)
    inversions = detect_priority_inversion(source, language)
    starv = detect_starvation(source, language)
    audit = synchronization_audit(source, language)

    # Deductions
    score -= len(deadlocks) * 25.0
    score -= len(livelocks) * 15.0
    score -= len(inversions) * 10.0
    score -= len(starv) * 8.0
    score -= len(audit.unreleased_locks) * 12.0

    # Missing sync
    lines = source.split('\n')
    thread_count = 0
    for pat in _THREAD_CREATE_PATTERNS.get(lang, []):
        thread_count += len(re.findall(pat, source))

    if thread_count > 0 and not audit.primitives:
        score -= 30.0

    # Bonus for RAII / safe patterns
    if lang in ("rust",):
        score += 5.0  # Rust's ownership model is inherently safer
    if re.search(r'(lock_guard|scoped_lock|unique_lock|MutexGuard|RwLockReadGuard)', source):
        score += 3.0
    if re.search(r'(channel|Chan|mpsc::channel|crossbeam::channel)', source):
        score += 3.0

    return max(0.0, min(100.0, score))


def analyze(source: str, language: str = "c") -> ConcurrencyReport:
    """Perform a comprehensive concurrency analysis of the given source code."""
    lang_enum = _resolve_language(language)

    deadlocks = detect_deadlocks(source, language)
    livelocks = detect_livelocks(source, language)
    inversions = detect_priority_inversion(source, language)
    starv = detect_starvation(source, language)
    audit = synchronization_audit(source, language)
    score = thread_safety_score(source, language)

    total_issues = len(deadlocks) + len(livelocks) + len(inversions) + len(starv)

    if total_issues == 0:
        summary = "No concurrency hazards detected."
    else:
        parts = []
        if deadlocks:
            parts.append(f"{len(deadlocks)} deadlock(s)")
        if livelocks:
            parts.append(f"{len(livelocks)} livelock(s)")
        if inversions:
            parts.append(f"{len(inversions)} priority inversion(s)")
        if starv:
            parts.append(f"{len(starv)} starvation risk(s)")
        summary = f"Found {total_issues} issue(s): {', '.join(parts)}."

    return ConcurrencyReport(
        language=lang_enum,
        deadlocks=deadlocks,
        livelocks=livelocks,
        priority_inversions=inversions,
        starvation_risks=starv,
        sync_audit=audit,
        thread_safety_score=score,
        summary=summary,
    )
