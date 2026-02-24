"""Transactional memory verification and analysis for concurrent programs."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re


class TMType(Enum):
    HTM = auto()
    STM = auto()
    HYBRID = auto()


class ConflictType(Enum):
    READ_WRITE = auto()
    WRITE_WRITE = auto()
    WRITE_READ = auto()
    FALSE_SHARING = auto()


class AbortReason(Enum):
    CONFLICT = auto()
    CAPACITY = auto()
    INTERRUPT = auto()
    NESTED_OVERFLOW = auto()
    EXPLICIT = auto()
    SYSCALL = auto()
    IO_IN_TX = auto()
    UNKNOWN = auto()


class TMPatternKind(Enum):
    LOCK_ELISION = auto()
    SPECULATIVE_LOCK = auto()
    RETRY_LOOP = auto()
    FALLBACK_LOCK = auto()
    NESTED_TX = auto()
    IRREVOCABLE = auto()


@dataclass
class TMRegion:
    start_line: int
    end_line: int
    tm_type: TMType
    read_set: Set[str] = field(default_factory=set)
    write_set: Set[str] = field(default_factory=set)
    nested_depth: int = 0
    has_fallback: bool = False
    can_abort: bool = True

    def __str__(self) -> str:
        rw = f"R={{{','.join(sorted(self.read_set))}}} W={{{','.join(sorted(self.write_set))}}}"
        return (f"TMRegion[{self.tm_type.name}](lines {self.start_line}-{self.end_line}, "
                f"{rw}, depth={self.nested_depth}, fallback={self.has_fallback})")


@dataclass
class TMConflict:
    region_a: TMRegion
    region_b: TMRegion
    conflict_type: ConflictType
    shared_vars: Set[str] = field(default_factory=set)
    description: str = ""
    severity: float = 0.0

    def __str__(self) -> str:
        return (f"TMConflict({self.conflict_type.name}, "
                f"vars={{{','.join(sorted(self.shared_vars))}}}, "
                f"severity={self.severity:.2f}: {self.description})")


@dataclass
class TMBoundary:
    suggested_start: int
    suggested_end: int
    variables: Set[str] = field(default_factory=set)
    reason: str = ""
    estimated_abort_rate: float = 0.0

    def __str__(self) -> str:
        return (f"TMBoundary(lines {self.suggested_start}-{self.suggested_end}, "
                f"vars={{{','.join(sorted(self.variables))}}}, "
                f"abort_rate={self.estimated_abort_rate:.2%}: {self.reason})")


@dataclass
class TMResult:
    source: str
    regions: List[TMRegion] = field(default_factory=list)
    conflicts: List[TMConflict] = field(default_factory=list)
    serializable: bool = True
    opacity_safe: bool = True
    issues: List[str] = field(default_factory=list)
    suggestions: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASS" if (self.serializable and self.opacity_safe and not self.issues) else "FAIL"
        return (f"TMResult({status}, {len(self.regions)} regions, "
                f"{len(self.conflicts)} conflicts, {len(self.issues)} issues)")


@dataclass
class TMComparison:
    htm_regions: List[TMRegion] = field(default_factory=list)
    stm_regions: List[TMRegion] = field(default_factory=list)
    htm_estimated_aborts: float = 0.0
    stm_estimated_aborts: float = 0.0
    htm_overhead_cycles: float = 0.0
    stm_overhead_cycles: float = 0.0
    recommendation: str = ""
    analysis: str = ""

    def __str__(self) -> str:
        return (f"TMComparison(HTM aborts={self.htm_estimated_aborts:.2%}, "
                f"STM aborts={self.stm_estimated_aborts:.2%}, "
                f"rec={self.recommendation})")


@dataclass
class TMPerfModel:
    regions: List[TMRegion] = field(default_factory=list)
    abort_rates: Dict[int, float] = field(default_factory=dict)
    retry_overhead: Dict[int, float] = field(default_factory=dict)
    contention_score: float = 0.0
    scalability_estimate: str = ""
    bottleneck_regions: List[int] = field(default_factory=list)

    def __str__(self) -> str:
        return (f"TMPerfModel(contention={self.contention_score:.2f}, "
                f"scalability={self.scalability_estimate}, "
                f"bottlenecks={self.bottleneck_regions})")


@dataclass
class TMPattern:
    kind: TMPatternKind
    line: int
    description: str
    code_snippet: str

    def __str__(self) -> str:
        return f"TMPattern({self.kind.name}, line {self.line}: {self.description})"


# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------

_TX_BEGIN_PATTERNS = [
    (re.compile(r'\b_xbegin\s*\(\s*\)'), TMType.HTM),
    (re.compile(r'\b__tm_begin\s*\('), TMType.HTM),
    (re.compile(r'\batomic\s*\{'), TMType.STM),
    (re.compile(r'\btransaction\s*\{'), TMType.STM),
    (re.compile(r'\bstd::atomic_transaction\s*\('), TMType.HYBRID),
    (re.compile(r'\bTM_BEGIN\b'), TMType.HTM),
    (re.compile(r'\bSTM_BEGIN\b'), TMType.STM),
]

_TX_END_PATTERNS = [
    re.compile(r'\b_xend\s*\(\s*\)'),
    re.compile(r'\b__tm_end\s*\('),
    re.compile(r'\bTM_END\b'),
    re.compile(r'\bSTM_END\b'),
]

_TX_ABORT_PATTERN = re.compile(r'\b_xabort\s*\(')

_SYSCALL_PATTERNS = [
    re.compile(r'\b(read|write|open|close|fork|exec|mmap|munmap|ioctl)\s*\('),
    re.compile(r'\bsyscall\s*\('),
    re.compile(r'\bsystem\s*\('),
]

_IO_PATTERNS = [
    re.compile(r'\b(printf|fprintf|puts|fputs|fwrite|fread|scanf|fscanf)\s*\('),
    re.compile(r'\b(cout|cin|cerr)\s*[<>]{2}'),
    re.compile(r'\bstd::(cout|cin|cerr)\s*[<>]{2}'),
]

_LOCK_BEGIN = [
    re.compile(r'\bpthread_mutex_lock\s*\(\s*&?\s*(\w+)'),
    re.compile(r'\bstd::lock_guard\s*<[^>]*>\s+(\w+)\s*\('),
    re.compile(r'\bstd::unique_lock\s*<[^>]*>\s+(\w+)\s*\('),
    re.compile(r'\bEnterCriticalSection\s*\('),
    re.compile(r'\bsynchronized\s*\('),
]

_LOCK_END = [
    re.compile(r'\bpthread_mutex_unlock\s*\(\s*&?\s*(\w+)'),
    re.compile(r'\bLeaveCriticalSection\s*\('),
]

_IDENTIFIER = re.compile(r'\b([a-zA-Z_]\w*)\b')
_ASSIGNMENT = re.compile(r'\b([a-zA-Z_]\w*)\s*(?:[+\-*/&|^]?=|(?:\+\+|--))')
_ARRAY_WRITE = re.compile(r'\b([a-zA-Z_]\w*)\s*\[')
_DEREF_WRITE = re.compile(r'\*\s*([a-zA-Z_]\w*)\s*=')

_KEYWORDS = frozenset({
    'if', 'else', 'while', 'for', 'do', 'switch', 'case', 'break', 'continue',
    'return', 'void', 'int', 'float', 'double', 'char', 'long', 'short',
    'unsigned', 'signed', 'const', 'static', 'volatile', 'struct', 'class',
    'typedef', 'enum', 'union', 'sizeof', 'true', 'false', 'NULL', 'nullptr',
    'auto', 'register', 'extern', 'inline', 'goto', 'default', 'bool',
    '_xbegin', '_xend', '_xabort', '__tm_begin', '__tm_end',
    'atomic', 'transaction', 'TM_BEGIN', 'TM_END', 'STM_BEGIN', 'STM_END',
    'pthread_mutex_lock', 'pthread_mutex_unlock', 'std',
})

_CACHE_LINE_BYTES = 64


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _parse_tm_regions(source: str) -> List[TMRegion]:
    """Find all transaction regions in *source*, returning a list of TMRegion."""
    lines = source.splitlines()
    regions: List[TMRegion] = []
    # Stack for nesting: each entry is (start_line, tm_type, depth)
    stack: List[Tuple[int, TMType, int]] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Check for tx begin
        for pat, tm_type in _TX_BEGIN_PATTERNS:
            if pat.search(stripped):
                depth = len(stack)
                stack.append((idx, tm_type, depth))
                break

        # Check for tx end (closing brace for atomic/transaction blocks)
        ended = False
        for pat in _TX_END_PATTERNS:
            if pat.search(stripped):
                ended = True
                break

        # Also detect closing brace matching an atomic/transaction block
        if not ended and stack and stripped == '}':
            top_type = stack[-1][1]
            if top_type in (TMType.STM, TMType.HYBRID):
                # Heuristic: count braces from the start line
                brace_depth = 0
                start_l = stack[-1][0]
                for j in range(start_l - 1, idx):
                    brace_depth += lines[j].count('{') - lines[j].count('}')
                if brace_depth <= 0:
                    ended = True

        if ended and stack:
            start_line, tm_type, depth = stack.pop()
            region = TMRegion(
                start_line=start_line,
                end_line=idx,
                tm_type=tm_type,
                nested_depth=depth,
                can_abort=True,
            )
            regions.append(region)

    # Handle unclosed regions — treat them as extending to end of source
    while stack:
        start_line, tm_type, depth = stack.pop()
        regions.append(TMRegion(
            start_line=start_line,
            end_line=len(lines),
            tm_type=tm_type,
            nested_depth=depth,
            can_abort=True,
        ))

    # Populate read/write sets and fallback info
    for region in regions:
        r_set, w_set = _extract_read_write_sets(source, region)
        region.read_set = r_set
        region.write_set = w_set
        # Check for fallback
        region.has_fallback = _region_has_fallback(source, region)

    return regions


def _extract_read_write_sets(source: str, region: TMRegion) -> Tuple[Set[str], Set[str]]:
    """Analyze variable accesses inside a transaction region."""
    lines = source.splitlines()
    start = max(0, region.start_line - 1)
    end = min(len(lines), region.end_line)
    body_lines = lines[start:end]

    write_set: Set[str] = set()
    all_ids: Set[str] = set()

    for line in body_lines:
        text = line.strip()
        # Skip comments
        if text.startswith('//') or text.startswith('/*') or text.startswith('*'):
            continue
        # Detect writes: assignments, increments, decrements
        for m in _ASSIGNMENT.finditer(text):
            var = m.group(1)
            if var not in _KEYWORDS:
                write_set.add(var)
        for m in _DEREF_WRITE.finditer(text):
            var = m.group(1)
            if var not in _KEYWORDS:
                write_set.add(var)
        # Collect all identifiers
        for m in _IDENTIFIER.finditer(text):
            ident = m.group(1)
            if ident not in _KEYWORDS and not ident.isupper():
                all_ids.add(ident)

    read_set = all_ids - write_set
    return read_set, write_set


def _region_has_fallback(source: str, region: TMRegion) -> bool:
    """Check if a TM region has a fallback path (retry loop or lock fallback)."""
    lines = source.splitlines()
    # Look a few lines before start and after end for fallback indicators
    search_start = max(0, region.start_line - 5)
    search_end = min(len(lines), region.end_line + 10)
    context = '\n'.join(lines[search_start:search_end])
    fallback_indicators = [
        'fallback', 'retry', 'RETRY', 'FALLBACK',
        'pthread_mutex_lock', 'lock_guard', 'acquire',
        '_XBEGIN_STARTED', 'status ==', 'while',
    ]
    return any(ind in context for ind in fallback_indicators)


def _detect_tm_patterns(source: str) -> List[TMPattern]:
    """Find transactional-memory usage patterns in *source*."""
    lines = source.splitlines()
    patterns: List[TMPattern] = []

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()

        # Lock elision: _xbegin before a lock acquire
        if '_xbegin' in stripped and idx < len(lines):
            lookahead = '\n'.join(lines[idx:min(idx + 5, len(lines))])
            if 'pthread_mutex' in lookahead or 'lock' in lookahead.lower():
                patterns.append(TMPattern(
                    kind=TMPatternKind.LOCK_ELISION,
                    line=idx,
                    description="Lock elision: speculative execution with lock fallback",
                    code_snippet=stripped,
                ))

        # Speculative lock
        if re.search(r'\bspeculat', stripped, re.IGNORECASE):
            patterns.append(TMPattern(
                kind=TMPatternKind.SPECULATIVE_LOCK,
                line=idx,
                description="Speculative lock pattern detected",
                code_snippet=stripped,
            ))

        # Retry loop
        if re.search(r'\b(while|for)\b.*\b(retry|_xbegin|TM_BEGIN)\b', stripped):
            patterns.append(TMPattern(
                kind=TMPatternKind.RETRY_LOOP,
                line=idx,
                description="Retry loop around transaction begin",
                code_snippet=stripped,
            ))

        # Fallback lock
        if re.search(r'\bfallback\b', stripped, re.IGNORECASE):
            patterns.append(TMPattern(
                kind=TMPatternKind.FALLBACK_LOCK,
                line=idx,
                description="Fallback lock path for failed transactions",
                code_snippet=stripped,
            ))

        # Nested transactions
        if re.search(r'\b(nested|inner)\s*(transaction|atomic|_xbegin)', stripped, re.IGNORECASE):
            patterns.append(TMPattern(
                kind=TMPatternKind.NESTED_TX,
                line=idx,
                description="Nested transaction detected",
                code_snippet=stripped,
            ))

        # Irrevocable
        if re.search(r'\birrevocable\b', stripped, re.IGNORECASE):
            patterns.append(TMPattern(
                kind=TMPatternKind.IRREVOCABLE,
                line=idx,
                description="Irrevocable transaction (cannot abort after this point)",
                code_snippet=stripped,
            ))

    return patterns


def _estimate_cache_footprint(region: TMRegion) -> int:
    """Estimate the number of cache lines touched by a TM region.

    Heuristic: each unique variable in the read+write set corresponds to roughly
    one 8-byte access; adjacent array elements may share a cache line.
    """
    all_vars = region.read_set | region.write_set
    # Approximate: each distinct variable is 8 bytes, potentially on its own cache line
    unique_count = len(all_vars)
    # Array-like names (contain digits or brackets context) may share lines
    array_vars = [v for v in all_vars if re.search(r'\d', v)]
    non_array = unique_count - len(array_vars)
    # Arrays: assume 2 elements per cache line on average
    array_lines = max(1, len(array_vars) // 2) if array_vars else 0
    return non_array + array_lines


def _check_nesting(regions: List[TMRegion]) -> List[str]:
    """Verify nesting correctness and return a list of issue descriptions."""
    issues: List[str] = []
    # Sort by start_line
    sorted_regions = sorted(regions, key=lambda r: r.start_line)

    for i, outer in enumerate(sorted_regions):
        for inner in sorted_regions[i + 1:]:
            # Check if inner is actually nested in outer
            if inner.start_line >= outer.start_line and inner.end_line <= outer.end_line:
                if outer.tm_type == TMType.HTM and inner.tm_type == TMType.HTM:
                    if inner.nested_depth > 7:
                        issues.append(
                            f"HTM nesting depth {inner.nested_depth} at line {inner.start_line} "
                            f"exceeds typical hardware limit (7-8 levels)")
                if outer.tm_type == TMType.STM and inner.tm_type == TMType.HTM:
                    issues.append(
                        f"HTM nested inside STM at line {inner.start_line} — "
                        f"mixing TM types in nested context is unsafe")
            # Overlapping but not properly nested
            elif (inner.start_line < outer.end_line and
                  inner.end_line > outer.end_line and
                  inner.start_line > outer.start_line):
                issues.append(
                    f"Overlapping (non-nested) transactions: "
                    f"lines {outer.start_line}-{outer.end_line} and "
                    f"lines {inner.start_line}-{inner.end_line}")
    return issues


def _find_lock_regions(source: str) -> List[Tuple[int, int, str, Set[str]]]:
    """Find mutex-protected critical sections.

    Returns list of (start_line, end_line, lock_name, variables_accessed).
    """
    lines = source.splitlines()
    results: List[Tuple[int, int, str, Set[str]]] = []
    lock_stack: List[Tuple[int, str]] = []  # (line, lock_name)

    for idx, line in enumerate(lines, start=1):
        stripped = line.strip()
        # Detect lock acquire
        for pat in _LOCK_BEGIN:
            m = pat.search(stripped)
            if m:
                lock_name = m.group(1) if m.lastindex and m.lastindex >= 1 else "unknown_lock"
                lock_stack.append((idx, lock_name))
                break

        # Detect lock release
        for pat in _LOCK_END:
            m = pat.search(stripped)
            if m and lock_stack:
                start_line, lock_name = lock_stack.pop()
                # Extract variables accessed in the critical section
                variables: Set[str] = set()
                for j in range(start_line, min(idx, len(lines))):
                    for im in _IDENTIFIER.finditer(lines[j]):
                        ident = im.group(1)
                        if ident not in _KEYWORDS:
                            variables.add(ident)
                results.append((start_line, idx, lock_name, variables))
                break

        # Handle scope-based locks (lock_guard) that end at closing brace
        if stripped == '}' and lock_stack:
            top_line, top_name = lock_stack[-1]
            # Check if this brace closes the scope containing the lock_guard
            brace_depth = 0
            for j in range(top_line - 1, idx):
                brace_depth += lines[j].count('{') - lines[j].count('}')
            if brace_depth <= 0:
                lock_stack.pop()
                variables = set()
                for j in range(top_line, min(idx, len(lines))):
                    for im in _IDENTIFIER.finditer(lines[j]):
                        ident = im.group(1)
                        if ident not in _KEYWORDS:
                            variables.add(ident)
                results.append((top_line, idx, top_name, variables))

    return results


def _check_unsafe_ops_in_region(source: str, region: TMRegion) -> List[str]:
    """Check for I/O, syscalls, or other non-transactional operations inside a tx."""
    lines = source.splitlines()
    start = max(0, region.start_line - 1)
    end = min(len(lines), region.end_line)
    issues: List[str] = []

    for line_idx in range(start, end):
        text = lines[line_idx].strip()
        lineno = line_idx + 1
        for pat in _SYSCALL_PATTERNS:
            if pat.search(text):
                issues.append(f"Syscall inside transaction at line {lineno}: {text}")
        for pat in _IO_PATTERNS:
            if pat.search(text):
                issues.append(f"I/O operation inside transaction at line {lineno}: {text}")
        if _TX_ABORT_PATTERN.search(text):
            issues.append(f"Explicit abort inside transaction at line {lineno}")

    return issues


def _compute_conflict_severity(region_a: TMRegion, region_b: TMRegion,
                               shared: Set[str], ctype: ConflictType) -> float:
    """Score severity of a conflict on a 0-1 scale."""
    base = 0.0
    # More shared variables → higher severity
    var_factor = min(1.0, len(shared) / 5.0)
    base += var_factor * 0.4

    # Write-write is worse than read-write
    type_weights = {
        ConflictType.WRITE_WRITE: 0.35,
        ConflictType.READ_WRITE: 0.25,
        ConflictType.WRITE_READ: 0.20,
        ConflictType.FALSE_SHARING: 0.15,
    }
    base += type_weights.get(ctype, 0.2)

    # Larger regions have higher conflict probability
    size_a = region_a.end_line - region_a.start_line
    size_b = region_b.end_line - region_b.start_line
    size_factor = min(1.0, (size_a + size_b) / 100.0)
    base += size_factor * 0.25

    return min(1.0, base)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def verify_tm_correctness(source: str) -> TMResult:
    """Parse source for transaction regions and verify correctness properties.

    Checks serializability, opacity, unsafe operations, fallback paths, and nesting.
    """
    regions = _parse_tm_regions(source)
    conflicts = detect_tm_conflicts(source)
    issues: List[str] = []
    suggestions: List[str] = []

    # Check for unsafe operations in every region
    for region in regions:
        region_issues = _check_unsafe_ops_in_region(source, region)
        issues.extend(region_issues)

    # Check nesting correctness
    nesting_issues = _check_nesting(regions)
    issues.extend(nesting_issues)

    # Check fallback paths
    for region in regions:
        if not region.has_fallback and region.tm_type == TMType.HTM:
            issues.append(
                f"HTM region at line {region.start_line} has no fallback path — "
                f"HTM can always abort, a non-transactional fallback is required")
            suggestions.append(
                f"Add a fallback lock path for HTM region at line {region.start_line}")

    # Check serializability: look for read-write conflicts across regions
    serializable = True
    for conflict in conflicts:
        if conflict.conflict_type in (ConflictType.READ_WRITE, ConflictType.WRITE_WRITE):
            if conflict.severity > 0.5:
                serializable = False
                break

    # Check opacity: can intermediate (uncommitted) state be observed?
    opacity_safe = True
    sorted_regions = sorted(regions, key=lambda r: r.start_line)
    for i, r1 in enumerate(sorted_regions):
        for r2 in sorted_regions[i + 1:]:
            # If r2 reads a variable that r1 writes, and they could execute concurrently,
            # intermediate state could be visible if r1 aborts after r2 reads.
            shared_rw = r1.write_set & r2.read_set
            if shared_rw and not r1.has_fallback:
                opacity_safe = False
                issues.append(
                    f"Opacity violation risk: region at line {r1.start_line} writes "
                    f"{shared_rw} which region at line {r2.start_line} reads — "
                    f"if the first aborts, an inconsistent snapshot may be observed")
                break
        if not opacity_safe:
            break

    # Check for empty transactions
    for region in regions:
        if not region.read_set and not region.write_set:
            issues.append(f"Empty transaction at line {region.start_line} (no data accesses)")
            suggestions.append(f"Remove empty transaction at line {region.start_line}")

    # Check cache capacity for HTM regions
    for region in regions:
        if region.tm_type == TMType.HTM:
            footprint = _estimate_cache_footprint(region)
            if footprint > 128:
                issues.append(
                    f"HTM region at line {region.start_line} touches ~{footprint} cache lines — "
                    f"likely exceeds L1 capacity, causing capacity aborts")
                suggestions.append(
                    f"Split large HTM region at line {region.start_line} into smaller transactions")

    # Suggest patterns
    patterns = _detect_tm_patterns(source)
    for pat in patterns:
        if pat.kind == TMPatternKind.RETRY_LOOP:
            suggestions.append(
                f"Retry loop at line {pat.line}: consider bounded retries with exponential backoff")

    if not regions:
        issues.append("No transactional memory regions found in source")

    return TMResult(
        source=source,
        regions=regions,
        conflicts=conflicts,
        serializable=serializable,
        opacity_safe=opacity_safe,
        issues=issues,
        suggestions=suggestions,
    )


def detect_tm_conflicts(source: str) -> List[TMConflict]:
    """Find pairwise conflicts between all TM regions in *source*."""
    regions = _parse_tm_regions(source)
    conflicts: List[TMConflict] = []

    for i, ra in enumerate(regions):
        for rb in regions[i + 1:]:
            # Read-Write conflict: ra reads what rb writes
            rw_shared = ra.read_set & rb.write_set
            if rw_shared:
                severity = _compute_conflict_severity(ra, rb, rw_shared, ConflictType.READ_WRITE)
                conflicts.append(TMConflict(
                    region_a=ra, region_b=rb,
                    conflict_type=ConflictType.READ_WRITE,
                    shared_vars=rw_shared,
                    description=(f"Region at line {ra.start_line} reads "
                                 f"{rw_shared} written by region at line {rb.start_line}"),
                    severity=severity,
                ))

            # Write-Read conflict: ra writes what rb reads
            wr_shared = ra.write_set & rb.read_set
            if wr_shared and wr_shared != rw_shared:
                severity = _compute_conflict_severity(ra, rb, wr_shared, ConflictType.WRITE_READ)
                conflicts.append(TMConflict(
                    region_a=ra, region_b=rb,
                    conflict_type=ConflictType.WRITE_READ,
                    shared_vars=wr_shared,
                    description=(f"Region at line {ra.start_line} writes "
                                 f"{wr_shared} read by region at line {rb.start_line}"),
                    severity=severity,
                ))

            # Write-Write conflict
            ww_shared = ra.write_set & rb.write_set
            if ww_shared:
                severity = _compute_conflict_severity(ra, rb, ww_shared, ConflictType.WRITE_WRITE)
                conflicts.append(TMConflict(
                    region_a=ra, region_b=rb,
                    conflict_type=ConflictType.WRITE_WRITE,
                    shared_vars=ww_shared,
                    description=(f"Both regions at lines {ra.start_line} and "
                                 f"{rb.start_line} write to {ww_shared}"),
                    severity=severity,
                ))

            # False sharing: variables that are different but may be on the same cache line
            a_only = (ra.read_set | ra.write_set) - (rb.read_set | rb.write_set)
            b_only = (rb.read_set | rb.write_set) - (ra.read_set | ra.write_set)
            if a_only and b_only:
                # Heuristic: variables with similar prefixes may be struct members
                for va in a_only:
                    for vb in b_only:
                        prefix_len = 0
                        for ca, cb in zip(va, vb):
                            if ca == cb:
                                prefix_len += 1
                            else:
                                break
                        if prefix_len >= 3 and prefix_len >= min(len(va), len(vb)) * 0.6:
                            fs_vars = {va, vb}
                            severity = _compute_conflict_severity(
                                ra, rb, fs_vars, ConflictType.FALSE_SHARING)
                            conflicts.append(TMConflict(
                                region_a=ra, region_b=rb,
                                conflict_type=ConflictType.FALSE_SHARING,
                                shared_vars=fs_vars,
                                description=(f"Potential false sharing between {va} and {vb} "
                                             f"(similar names suggest adjacent memory layout)"),
                                severity=severity * 0.6,
                            ))

    # Sort by severity descending
    conflicts.sort(key=lambda c: c.severity, reverse=True)
    return conflicts


def suggest_tm_boundaries(source: str) -> List[TMBoundary]:
    """Suggest converting lock-based critical sections to transactions."""
    lock_regions = _find_lock_regions(source)
    boundaries: List[TMBoundary] = []

    for start, end, lock_name, variables in lock_regions:
        region_size = end - start
        var_count = len(variables)

        # Estimate abort rate based on region size and variable count
        # Larger regions: higher capacity abort probability
        capacity_factor = min(1.0, region_size / 50.0) * 0.3
        # More variables: higher conflict probability
        conflict_factor = min(1.0, var_count / 20.0) * 0.3
        # Base abort rate for any transaction
        base_rate = 0.02
        estimated_abort = base_rate + capacity_factor + conflict_factor

        reason_parts = [f"Lock '{lock_name}' protects {var_count} variables over {region_size} lines"]

        if region_size > 30:
            reason_parts.append(
                "Consider splitting: region is large, which increases capacity abort risk")
            estimated_abort = min(0.95, estimated_abort * 1.5)
        elif region_size < 5:
            reason_parts.append("Good candidate for HTM: small critical section")
            estimated_abort *= 0.5

        if var_count > 15:
            reason_parts.append("Many variables accessed — consider partitioning data")

        boundaries.append(TMBoundary(
            suggested_start=start,
            suggested_end=end,
            variables=variables,
            reason='; '.join(reason_parts),
            estimated_abort_rate=min(1.0, estimated_abort),
        ))

    # If no lock regions, scan for coarse patterns (global data accesses)
    if not boundaries:
        lines = source.splitlines()
        # Look for shared global writes that might benefit from transactions
        global_writes: Dict[str, List[int]] = {}
        for idx, line in enumerate(lines, start=1):
            for m in _ASSIGNMENT.finditer(line):
                var = m.group(1)
                if var not in _KEYWORDS and not var[0].isupper():
                    global_writes.setdefault(var, []).append(idx)

        # Variables written from multiple places are candidates
        for var, write_lines in global_writes.items():
            if len(write_lines) >= 2:
                for wl in write_lines:
                    boundaries.append(TMBoundary(
                        suggested_start=max(1, wl - 1),
                        suggested_end=wl + 1,
                        variables={var},
                        reason=f"Variable '{var}' written at {len(write_lines)} locations — "
                               f"consider protecting with a transaction",
                        estimated_abort_rate=0.05,
                    ))

    # Sort by estimated abort rate (best candidates first)
    boundaries.sort(key=lambda b: b.estimated_abort_rate)
    return boundaries


def htm_vs_stm_analysis(source: str) -> TMComparison:
    """Compare HTM and STM suitability for each region in *source*."""
    regions = _parse_tm_regions(source)

    if not regions:
        # Analyze lock regions as potential TM candidates
        lock_regions = _find_lock_regions(source)
        for start, end, _, variables in lock_regions:
            regions.append(TMRegion(
                start_line=start, end_line=end,
                tm_type=TMType.HYBRID,
                read_set=variables, write_set=set(),
                nested_depth=0, has_fallback=True,
            ))

    htm_regions: List[TMRegion] = []
    stm_regions: List[TMRegion] = []
    total_htm_aborts = 0.0
    total_stm_aborts = 0.0
    total_htm_cycles = 0.0
    total_stm_cycles = 0.0

    for region in regions:
        cache_footprint = _estimate_cache_footprint(region)
        region_size = region.end_line - region.start_line + 1
        var_count = len(region.read_set) + len(region.write_set)
        has_unsafe = bool(_check_unsafe_ops_in_region(source, region))

        # HTM analysis
        htm_copy = TMRegion(
            start_line=region.start_line, end_line=region.end_line,
            tm_type=TMType.HTM,
            read_set=set(region.read_set), write_set=set(region.write_set),
            nested_depth=region.nested_depth,
            has_fallback=region.has_fallback, can_abort=True,
        )

        # HTM abort rate: capacity aborts dominate for large footprints
        htm_capacity_abort = min(0.9, cache_footprint / 256.0)
        htm_conflict_abort = min(0.5, var_count / 30.0) * 0.3
        htm_syscall_abort = 1.0 if has_unsafe else 0.0
        htm_abort = min(1.0, htm_capacity_abort + htm_conflict_abort + htm_syscall_abort)

        # HTM overhead: ~5 cycles begin + ~5 cycles end per attempt
        htm_attempts = 1.0 / max(0.01, 1.0 - htm_abort) if htm_abort < 1.0 else 100.0
        htm_cycle = htm_attempts * 10.0  # 10 cycles per attempt
        if htm_abort >= 0.9:
            htm_cycle += 500.0  # fallback lock overhead

        htm_regions.append(htm_copy)
        total_htm_aborts += htm_abort
        total_htm_cycles += htm_cycle

        # STM analysis
        stm_copy = TMRegion(
            start_line=region.start_line, end_line=region.end_line,
            tm_type=TMType.STM,
            read_set=set(region.read_set), write_set=set(region.write_set),
            nested_depth=region.nested_depth,
            has_fallback=True, can_abort=True,
        )

        # STM abort rate: mainly conflict-driven, no capacity limit
        stm_conflict_abort = min(0.6, var_count / 20.0) * 0.4
        stm_validation_abort = min(0.3, region_size / 100.0) * 0.2
        stm_abort = min(1.0, stm_conflict_abort + stm_validation_abort)

        # STM overhead: 100-500 cycles for logging per variable
        stm_log_overhead = var_count * 50.0  # ~50 cycles per variable for read/write logging
        stm_validation = len(region.read_set) * 20.0  # validation overhead per read
        stm_cycle = stm_log_overhead + stm_validation + 100.0  # base begin/commit cost

        stm_regions.append(stm_copy)
        total_stm_aborts += stm_abort
        total_stm_cycles += stm_cycle

    n = max(1, len(regions))
    avg_htm_aborts = total_htm_aborts / n
    avg_stm_aborts = total_stm_aborts / n
    avg_htm_cycles = total_htm_cycles / n
    avg_stm_cycles = total_stm_cycles / n

    # Generate recommendation
    if not regions:
        recommendation = "NONE"
        analysis = "No transactional regions or lock-based critical sections found."
    elif avg_htm_aborts > 0.7:
        recommendation = "STM"
        analysis = (f"HTM estimated abort rate is too high ({avg_htm_aborts:.0%}). "
                    f"Regions are likely too large or contain unsafe operations. "
                    f"STM provides reliable execution with {avg_stm_cycles:.0f} cycle overhead.")
    elif avg_stm_cycles > avg_htm_cycles * 5:
        recommendation = "HTM"
        analysis = (f"HTM is significantly faster ({avg_htm_cycles:.0f} vs "
                    f"{avg_stm_cycles:.0f} cycles). Low abort rate ({avg_htm_aborts:.0%}) "
                    f"makes hardware transactions efficient.")
    elif avg_htm_aborts < 0.15 and avg_htm_cycles < 100:
        recommendation = "HTM"
        analysis = (f"Regions are small and well-suited for HTM. "
                    f"Expected {avg_htm_aborts:.0%} abort rate with only "
                    f"{avg_htm_cycles:.0f} cycle overhead.")
    else:
        recommendation = "HYBRID"
        analysis = (f"Mixed workload: HTM abort rate {avg_htm_aborts:.0%}, "
                    f"STM overhead {avg_stm_cycles:.0f} cycles. "
                    f"Use HTM with STM fallback for best performance.")

    return TMComparison(
        htm_regions=htm_regions,
        stm_regions=stm_regions,
        htm_estimated_aborts=avg_htm_aborts,
        stm_estimated_aborts=avg_stm_aborts,
        htm_overhead_cycles=avg_htm_cycles,
        stm_overhead_cycles=avg_stm_cycles,
        recommendation=recommendation,
        analysis=analysis,
    )


def tm_performance_model(source: str) -> TMPerfModel:
    """Model performance characteristics of transactional regions in *source*."""
    regions = _parse_tm_regions(source)

    if not regions:
        return TMPerfModel(
            regions=[],
            abort_rates={},
            retry_overhead={},
            contention_score=0.0,
            scalability_estimate="serial",
            bottleneck_regions=[],
        )

    abort_rates: Dict[int, float] = {}
    retry_overhead: Dict[int, float] = {}
    bottleneck_indices: List[int] = []

    # Compute shared variable frequency across all regions
    global_write_freq: Dict[str, int] = {}
    for region in regions:
        for var in region.write_set:
            global_write_freq[var] = global_write_freq.get(var, 0) + 1

    total_contention = 0.0

    for i, region in enumerate(regions):
        region_size = region.end_line - region.start_line + 1
        var_count = len(region.read_set) + len(region.write_set)
        cache_footprint = _estimate_cache_footprint(region)

        # Abort rate model
        # Component 1: capacity (HTM-specific)
        cap_abort = min(0.8, cache_footprint / 200.0) if region.tm_type == TMType.HTM else 0.0
        # Component 2: conflict (shared writes)
        conflict_vars = sum(1 for v in region.write_set if global_write_freq.get(v, 0) > 1)
        conflict_vars += sum(1 for v in region.read_set if global_write_freq.get(v, 0) > 0)
        conflict_abort = min(0.7, conflict_vars / max(1, var_count) * 0.5)
        # Component 3: size-based (longer tx = more conflict window)
        size_abort = min(0.3, region_size / 80.0) * 0.2

        total_abort = min(0.95, cap_abort + conflict_abort + size_abort)
        abort_rates[i] = total_abort

        # Retry overhead: expected retries = 1/(1-abort_rate), each retry costs region_size cycles
        expected_retries = 1.0 / max(0.05, 1.0 - total_abort) - 1.0
        retry_cost = expected_retries * region_size * 10.0  # ~10 cycles per line
        retry_overhead[i] = retry_cost

        # Contention contribution
        region_contention = conflict_abort * (1.0 + expected_retries)
        total_contention += region_contention

        # Bottleneck: high abort rate or high contention
        if total_abort > 0.4 or region_contention > 1.5:
            bottleneck_indices.append(i)

    # Normalize contention score to 0-1
    contention_score = min(1.0, total_contention / max(1, len(regions)))

    # Scalability estimate
    if contention_score > 0.8:
        scalability = "serial"
    elif contention_score > 0.5:
        scalability = "limited"
    elif contention_score > 0.2:
        scalability = "linear"
    else:
        scalability = "superlinear"

    return TMPerfModel(
        regions=regions,
        abort_rates=abort_rates,
        retry_overhead=retry_overhead,
        contention_score=contention_score,
        scalability_estimate=scalability,
        bottleneck_regions=bottleneck_indices,
    )
