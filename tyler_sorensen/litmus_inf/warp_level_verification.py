"""GPU warp-level verification for CUDA kernel primitives.

Analyzes warp shuffle, vote, reduce, cooperative groups, tensor core,
and divergence patterns in CUDA source code via regex-based parsing.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Dict, Optional, Tuple, Set
import re


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class WarpPrimitive(Enum):
    SHUFFLE_SYNC = auto()
    SHUFFLE_UP = auto()
    SHUFFLE_DOWN = auto()
    SHUFFLE_XOR = auto()
    VOTE_SYNC = auto()
    VOTE_ALL = auto()
    VOTE_ANY = auto()
    VOTE_BALLOT = auto()
    REDUCE_ADD = auto()
    REDUCE_MIN = auto()
    REDUCE_MAX = auto()
    REDUCE_AND = auto()
    REDUCE_OR = auto()
    REDUCE_XOR = auto()
    MATCH_ANY = auto()
    MATCH_ALL = auto()

    def __str__(self) -> str:
        return self.name


class SyncIssueType(Enum):
    MISSING_MASK = auto()
    PARTIAL_WARP = auto()
    DIVERGENT_WARP = auto()
    INACTIVE_THREAD = auto()
    MASK_MISMATCH = auto()
    MISSING_SYNC = auto()
    RACE_CONDITION = auto()
    DEADLOCK = auto()

    def __str__(self) -> str:
        return self.name


class DivergenceKind(Enum):
    BRANCH_DIVERGENCE = auto()
    LOOP_DIVERGENCE = auto()
    RETURN_DIVERGENCE = auto()
    SWITCH_DIVERGENCE = auto()
    INDIRECT_CALL = auto()

    def __str__(self) -> str:
        return self.name


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class WarpOp:
    line: int
    primitive: WarpPrimitive
    mask_expr: str
    source_lane: Optional[int]
    value_expr: str
    in_divergent_branch: bool

    def __str__(self) -> str:
        loc = f"line {self.line}"
        div = " [DIVERGENT]" if self.in_divergent_branch else ""
        return f"WarpOp({self.primitive}, mask={self.mask_expr}, lane={self.source_lane}{div}) @ {loc}"


@dataclass
class ShuffleIssue:
    line: int
    primitive: WarpPrimitive
    issue_type: SyncIssueType
    description: str
    fix: str
    severity: str

    def __str__(self) -> str:
        return f"[{self.severity}] ShuffleIssue({self.issue_type}) @ line {self.line}: {self.description}"


@dataclass
class WarpShuffleResult:
    kernel_name: str
    shuffles_found: List[WarpOp]
    issues: List[ShuffleIssue]
    lane_coverage: Dict[int, bool]
    safe: bool

    def __str__(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        covered = sum(1 for v in self.lane_coverage.values() if v)
        return (f"WarpShuffleResult({self.kernel_name}): {status}, "
                f"{len(self.shuffles_found)} shuffles, {len(self.issues)} issues, "
                f"{covered}/32 lanes covered")


@dataclass
class VoteIssue:
    line: int
    primitive: WarpPrimitive
    issue_type: SyncIssueType
    description: str
    fix: str

    def __str__(self) -> str:
        return f"VoteIssue({self.issue_type}) @ line {self.line}: {self.description}"


@dataclass
class WarpVoteResult:
    kernel_name: str
    votes_found: List[WarpOp]
    issues: List[VoteIssue]
    safe: bool

    def __str__(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        return (f"WarpVoteResult({self.kernel_name}): {status}, "
                f"{len(self.votes_found)} votes, {len(self.issues)} issues")


@dataclass
class ReduceIssue:
    line: int
    primitive: WarpPrimitive
    issue_type: SyncIssueType
    description: str
    fix: str

    def __str__(self) -> str:
        return f"ReduceIssue({self.issue_type}) @ line {self.line}: {self.description}"


@dataclass
class WarpReduceResult:
    kernel_name: str
    reduces_found: List[WarpOp]
    issues: List[ReduceIssue]
    safe: bool

    def __str__(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        return (f"WarpReduceResult({self.kernel_name}): {status}, "
                f"{len(self.reduces_found)} reduces, {len(self.issues)} issues")


@dataclass
class CoopGroupIssue:
    line: int
    issue_type: SyncIssueType
    description: str
    fix: str
    group_type: str

    def __str__(self) -> str:
        return f"CoopGroupIssue({self.issue_type}, {self.group_type}) @ line {self.line}: {self.description}"


@dataclass
class CoopGroupResult:
    kernel_name: str
    groups_found: int
    issues: List[CoopGroupIssue]
    sync_points: int
    safe: bool

    def __str__(self) -> str:
        status = "SAFE" if self.safe else "UNSAFE"
        return (f"CoopGroupResult({self.kernel_name}): {status}, "
                f"{self.groups_found} groups, {self.sync_points} sync points, "
                f"{len(self.issues)} issues")


@dataclass
class TensorCoreIssue:
    line: int
    issue_type: SyncIssueType
    description: str
    fix: str

    def __str__(self) -> str:
        return f"TensorCoreIssue({self.issue_type}) @ line {self.line}: {self.description}"


@dataclass
class TensorCoreResult:
    kernel_name: str
    mma_ops: int
    issues: List[TensorCoreIssue]
    sync_correct: bool

    def __str__(self) -> str:
        status = "SYNC_OK" if self.sync_correct else "SYNC_ERROR"
        return (f"TensorCoreResult({self.kernel_name}): {status}, "
                f"{self.mma_ops} MMA ops, {len(self.issues)} issues")


@dataclass
class DivergencePoint:
    line: int
    kind: DivergenceKind
    condition: str
    reconvergence_line: Optional[int]
    affected_threads: str

    def __str__(self) -> str:
        reconv = f" -> reconverge@{self.reconvergence_line}" if self.reconvergence_line else ""
        return (f"DivergencePoint({self.kind}, \"{self.condition}\") @ line {self.line}"
                f"{reconv}, affects {self.affected_threads}")


@dataclass
class DivergenceReport:
    kernel_name: str
    divergence_points: List[DivergencePoint]
    warp_efficiency: float
    serialized_instructions: int
    total_instructions: int

    def __str__(self) -> str:
        eff = f"{self.warp_efficiency:.1%}"
        return (f"DivergenceReport({self.kernel_name}): efficiency={eff}, "
                f"{len(self.divergence_points)} divergence points, "
                f"{self.serialized_instructions}/{self.total_instructions} serialized")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

_KERNEL_RE = re.compile(
    r'(?:__global__|__device__)\s+[\w\s\*&:<>,]+?\s+(\w+)\s*\(([^)]*)\)\s*\{',
    re.DOTALL,
)


def _parse_cuda_kernel(source: str) -> List[Tuple[str, str]]:
    """Extract kernel functions from CUDA source.

    Returns list of (kernel_name, kernel_body) tuples.
    """
    kernels: List[Tuple[str, str]] = []
    for m in _KERNEL_RE.finditer(source):
        name = m.group(1)
        start = m.end() - 1  # the opening brace
        depth = 0
        body_end = start
        for i in range(start, len(source)):
            if source[i] == '{':
                depth += 1
            elif source[i] == '}':
                depth -= 1
                if depth == 0:
                    body_end = i + 1
                    break
        kernels.append((name, source[start:body_end]))
    return kernels


def _find_divergent_branches(kernel: str) -> List[Tuple[int, str]]:
    """Find branches dependent on threadIdx / lane id.

    Returns list of (line_number, condition_text).
    """
    thread_dep_re = re.compile(
        r'\b(if|while|for)\s*\(([^)]*(?:threadIdx|laneid|lane_id|tid\b)[^)]*)\)',
        re.IGNORECASE,
    )
    results: List[Tuple[int, str]] = []
    for lineno, line in enumerate(kernel.splitlines(), start=1):
        m = thread_dep_re.search(line)
        if m:
            results.append((lineno, m.group(2).strip()))
    return results


def _check_mask_validity(mask_expr: str) -> Tuple[bool, str]:
    """Validate a warp mask expression.

    Returns (is_valid, reason).
    Full-warp masks (0xffffffff, ~0, 0xFFFFFFFF, FULL_MASK) are always valid.
    Hex literals are checked for 32-bit range.
    """
    stripped = mask_expr.strip()
    full_masks = {"0xffffffff", "0xFFFFFFFF", "~0", "~0u", "~0U",
                  "0xFFFFFFFFu", "0xffffffffu", "FULL_MASK", "__activemask()"}
    if stripped in full_masks:
        return True, "full warp mask"
    hex_match = re.match(r'^0[xX]([0-9a-fA-F]+)[uU]?$', stripped)
    if hex_match:
        val = int(hex_match.group(1), 16)
        if val == 0:
            return False, "mask is zero — no threads participate"
        if val > 0xFFFFFFFF:
            return False, "mask exceeds 32-bit range"
        active = bin(val).count('1')
        if active < 32:
            return True, f"partial mask: {active}/32 lanes active"
        return True, "full warp mask"
    if re.match(r'^\d+[uU]?$', stripped):
        val = int(re.sub(r'[uU]$', '', stripped))
        if val == 0:
            return False, "mask is zero — no threads participate"
        if val > 0xFFFFFFFF:
            return False, "mask exceeds 32-bit range"
        active = bin(val).count('1')
        return True, f"decimal mask: {active}/32 lanes active"
    # variable or expression — cannot statically verify
    return True, f"dynamic mask expression '{stripped}' — verify at runtime"


def _find_reconvergence(source: str, branch_line: int) -> Optional[int]:
    """Find the reconvergence point after a divergent branch.

    Scans for the matching closing brace of an if-block, then checks for
    an else block.  The reconvergence point is the line after the last
    closing brace of the if/else chain.
    """
    lines = source.splitlines()
    if branch_line < 1 or branch_line > len(lines):
        return None
    depth = 0
    found_open = False
    last_close_line: Optional[int] = None
    for idx in range(branch_line - 1, len(lines)):
        for ch in lines[idx]:
            if ch == '{':
                depth += 1
                found_open = True
            elif ch == '}':
                depth -= 1
                if found_open and depth == 0:
                    last_close_line = idx + 1  # 1-based
                    # check if an else follows
                    rest = ''.join(lines[idx:idx + 3]) if idx + 3 <= len(lines) else ''.join(lines[idx:])
                    if re.search(r'}\s*else\s*\{', rest) or re.search(r'}\s*else\s+if\s*\(', rest):
                        found_open = True
                        depth = 0
                        continue
                    return last_close_line + 1 if last_close_line < len(lines) else last_close_line
    return last_close_line


def _line_number_in_kernel(kernel_body: str, match_start: int) -> int:
    """Translate a character offset inside *kernel_body* to a 1-based line number."""
    return kernel_body[:match_start].count('\n') + 1


def _is_inside_divergent(kernel_body: str, char_offset: int,
                         divergent_branches: List[Tuple[int, str]]) -> bool:
    """Return True when *char_offset* sits inside a known divergent branch."""
    op_line = _line_number_in_kernel(kernel_body, char_offset)
    for branch_line, _ in divergent_branches:
        reconv = _find_reconvergence(kernel_body, branch_line)
        if reconv is None:
            reconv = branch_line + 30  # heuristic
        if branch_line <= op_line <= reconv:
            return True
    return False


def _extract_kernel_name(source: str) -> str:
    """Return the first kernel name found, or 'unknown'."""
    kernels = _parse_cuda_kernel(source)
    return kernels[0][0] if kernels else "unknown"


def _mask_to_lane_coverage(mask_expr: str) -> Dict[int, bool]:
    """Convert a mask expression to per-lane coverage dict (lanes 0-31)."""
    coverage: Dict[int, bool] = {i: False for i in range(32)}
    stripped = mask_expr.strip()
    full_masks = {"0xffffffff", "0xFFFFFFFF", "~0", "~0u", "~0U",
                  "0xFFFFFFFFu", "0xffffffffu", "FULL_MASK", "__activemask()"}
    if stripped in full_masks:
        return {i: True for i in range(32)}
    hex_match = re.match(r'^0[xX]([0-9a-fA-F]+)[uU]?$', stripped)
    val: Optional[int] = None
    if hex_match:
        val = int(hex_match.group(1), 16)
    elif re.match(r'^\d+[uU]?$', stripped):
        val = int(re.sub(r'[uU]$', '', stripped))
    if val is not None:
        for lane in range(32):
            coverage[lane] = bool(val & (1 << lane))
    else:
        # dynamic — assume full participation
        coverage = {i: True for i in range(32)}
    return coverage


# ---------------------------------------------------------------------------
# Shuffle verification
# ---------------------------------------------------------------------------

_SHUFFLE_PATTERNS: List[Tuple[re.Pattern, WarpPrimitive]] = [
    (re.compile(r'__shfl_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)'), WarpPrimitive.SHUFFLE_SYNC),
    (re.compile(r'__shfl_up_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)'), WarpPrimitive.SHUFFLE_UP),
    (re.compile(r'__shfl_down_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)'), WarpPrimitive.SHUFFLE_DOWN),
    (re.compile(r'__shfl_xor_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,\)]+)'), WarpPrimitive.SHUFFLE_XOR),
]


def verify_warp_shuffle(kernel: str) -> WarpShuffleResult:
    """Parse and verify all warp shuffle operations in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    divergent = _find_divergent_branches(kernel)
    ops: List[WarpOp] = []
    issues: List[ShuffleIssue] = []
    lane_coverage: Dict[int, bool] = {i: False for i in range(32)}
    seen_masks: List[str] = []

    for pat, prim in _SHUFFLE_PATTERNS:
        for m in pat.finditer(kernel):
            mask_expr = m.group(1).strip()
            value_expr = m.group(2).strip()
            lane_or_delta = m.group(3).strip()
            lineno = _line_number_in_kernel(kernel, m.start())
            in_div = _is_inside_divergent(kernel, m.start(), divergent)

            # parse source lane
            src_lane: Optional[int] = None
            lane_match = re.match(r'^(\d+)$', lane_or_delta)
            if lane_match:
                src_lane = int(lane_match.group(1))

            op = WarpOp(lineno, prim, mask_expr, src_lane, value_expr, in_div)
            ops.append(op)
            seen_masks.append(mask_expr)

            # update lane coverage
            cov = _mask_to_lane_coverage(mask_expr)
            for lane, active in cov.items():
                if active:
                    lane_coverage[lane] = True

            # --- issue checks ---
            valid, reason = _check_mask_validity(mask_expr)
            if not valid:
                issues.append(ShuffleIssue(
                    lineno, prim, SyncIssueType.MISSING_MASK,
                    f"Invalid mask {mask_expr}: {reason}",
                    "Use 0xffffffff for full-warp shuffle or a valid sub-mask.",
                    "error",
                ))
            elif "partial" in reason:
                issues.append(ShuffleIssue(
                    lineno, prim, SyncIssueType.PARTIAL_WARP,
                    f"Partial warp shuffle with mask {mask_expr}: {reason}",
                    "Ensure all intended lanes are set in the mask.",
                    "warning",
                ))

            if in_div:
                issues.append(ShuffleIssue(
                    lineno, prim, SyncIssueType.DIVERGENT_WARP,
                    f"Shuffle {prim} at line {lineno} is inside a threadIdx-dependent branch.",
                    "Move shuffle outside divergent branch or use __activemask().",
                    "error",
                ))

            if src_lane is not None:
                if prim == WarpPrimitive.SHUFFLE_SYNC and not (0 <= src_lane <= 31):
                    issues.append(ShuffleIssue(
                        lineno, prim, SyncIssueType.INACTIVE_THREAD,
                        f"Source lane {src_lane} is out of valid range 0-31.",
                        "Use a lane index between 0 and 31.",
                        "error",
                    ))
                if prim in (WarpPrimitive.SHUFFLE_UP, WarpPrimitive.SHUFFLE_DOWN):
                    if src_lane > 31:
                        issues.append(ShuffleIssue(
                            lineno, prim, SyncIssueType.INACTIVE_THREAD,
                            f"Delta {src_lane} for {prim} exceeds warp size.",
                            "Keep delta within 0-31.",
                            "error",
                        ))

    # check mask consistency across shuffle chain
    unique_masks = set(seen_masks)
    if len(unique_masks) > 1:
        for op in ops:
            if op.mask_expr != seen_masks[0]:
                issues.append(ShuffleIssue(
                    op.line, op.primitive, SyncIssueType.MASK_MISMATCH,
                    f"Mask {op.mask_expr} differs from first shuffle mask {seen_masks[0]}.",
                    "Use a consistent mask across shuffle chains.",
                    "warning",
                ))

    safe = len(issues) == 0 or all(i.severity == "warning" for i in issues)
    return WarpShuffleResult(kernel_name, ops, issues, lane_coverage, safe)


# ---------------------------------------------------------------------------
# Vote verification
# ---------------------------------------------------------------------------

_VOTE_PATTERNS: List[Tuple[re.Pattern, WarpPrimitive]] = [
    (re.compile(r'__all_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.VOTE_ALL),
    (re.compile(r'__any_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.VOTE_ANY),
    (re.compile(r'__ballot_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.VOTE_BALLOT),
]


def verify_warp_vote(kernel: str) -> WarpVoteResult:
    """Parse and verify all warp vote operations in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    divergent = _find_divergent_branches(kernel)
    ops: List[WarpOp] = []
    issues: List[VoteIssue] = []

    for pat, prim in _VOTE_PATTERNS:
        for m in pat.finditer(kernel):
            mask_expr = m.group(1).strip()
            value_expr = m.group(2).strip()
            lineno = _line_number_in_kernel(kernel, m.start())
            in_div = _is_inside_divergent(kernel, m.start(), divergent)

            op = WarpOp(lineno, prim, mask_expr, None, value_expr, in_div)
            ops.append(op)

            valid, reason = _check_mask_validity(mask_expr)
            if not valid:
                issues.append(VoteIssue(
                    lineno, prim, SyncIssueType.MISSING_MASK,
                    f"Invalid mask {mask_expr}: {reason}",
                    "Use 0xffffffff or a valid mask covering all participating threads.",
                ))

            if in_div:
                issues.append(VoteIssue(
                    lineno, prim, SyncIssueType.DIVERGENT_WARP,
                    f"Vote {prim} at line {lineno} is inside a divergent branch.",
                    "Move vote outside divergent code or adjust mask to match active threads.",
                ))

            # warn about partial masks with vote_all / vote_any
            if "partial" in reason and prim in (WarpPrimitive.VOTE_ALL, WarpPrimitive.VOTE_ANY):
                issues.append(VoteIssue(
                    lineno, prim, SyncIssueType.PARTIAL_WARP,
                    f"{prim} uses partial mask {mask_expr}; result only reflects subset of warp.",
                    "Ensure the mask covers all threads whose predicate matters.",
                ))

    safe = len(issues) == 0
    return WarpVoteResult(kernel_name, ops, issues, safe)


# ---------------------------------------------------------------------------
# Reduce verification
# ---------------------------------------------------------------------------

_REDUCE_PATTERNS: List[Tuple[re.Pattern, WarpPrimitive]] = [
    (re.compile(r'__reduce_add_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_ADD),
    (re.compile(r'__reduce_min_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_MIN),
    (re.compile(r'__reduce_max_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_MAX),
    (re.compile(r'__reduce_and_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_AND),
    (re.compile(r'__reduce_or_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_OR),
    (re.compile(r'__reduce_xor_sync\s*\(\s*([^,]+),\s*([^)]+)\)'), WarpPrimitive.REDUCE_XOR),
]

_FLOAT_TYPE_RE = re.compile(r'\b(float|double|half|__half)\b')


def verify_warp_reduce(kernel: str) -> WarpReduceResult:
    """Parse and verify all warp reduce operations in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    divergent = _find_divergent_branches(kernel)
    ops: List[WarpOp] = []
    issues: List[ReduceIssue] = []

    for pat, prim in _REDUCE_PATTERNS:
        for m in pat.finditer(kernel):
            mask_expr = m.group(1).strip()
            value_expr = m.group(2).strip()
            lineno = _line_number_in_kernel(kernel, m.start())
            in_div = _is_inside_divergent(kernel, m.start(), divergent)

            op = WarpOp(lineno, prim, mask_expr, None, value_expr, in_div)
            ops.append(op)

            valid, reason = _check_mask_validity(mask_expr)
            if not valid:
                issues.append(ReduceIssue(
                    lineno, prim, SyncIssueType.MISSING_MASK,
                    f"Invalid mask {mask_expr}: {reason}",
                    "Use 0xffffffff for full-warp reduce.",
                ))

            if in_div:
                issues.append(ReduceIssue(
                    lineno, prim, SyncIssueType.DIVERGENT_WARP,
                    f"Reduce {prim} at line {lineno} inside divergent branch.",
                    "Move reduce outside divergent code or guard with __activemask().",
                ))

            # bitwise ops on floats
            if prim in (WarpPrimitive.REDUCE_AND, WarpPrimitive.REDUCE_OR,
                        WarpPrimitive.REDUCE_XOR):
                # look backward for declaration of value_expr
                var_name = re.split(r'[\[\(+\-*/]', value_expr)[0].strip()
                decl_re = re.compile(rf'\b(float|double|half|__half)\s+{re.escape(var_name)}\b')
                if decl_re.search(kernel):
                    issues.append(ReduceIssue(
                        lineno, prim, SyncIssueType.MASK_MISMATCH,
                        f"Bitwise reduce {prim} applied to floating-point variable '{var_name}'.",
                        "Use integer types with bitwise warp reductions.",
                    ))

    safe = len(issues) == 0
    return WarpReduceResult(kernel_name, ops, issues, safe)


# ---------------------------------------------------------------------------
# Cooperative groups verification
# ---------------------------------------------------------------------------

_CG_GROUP_RE = re.compile(
    r'(cooperative_groups::(?:thread_block|grid_group|tiled_partition<\d+>))\s+(\w+)'
)
_CG_THIS_RE = re.compile(
    r'(?:this_thread_block|this_grid)\s*\(\s*\)'
)
_CG_SYNC_RE = re.compile(r'(\w+)\.sync\s*\(\s*\)')
_CG_TILE_RE = re.compile(r'tiled_partition<(\d+)>')


def verify_cooperative_groups(kernel: str) -> CoopGroupResult:
    """Parse and verify cooperative group usage in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    divergent = _find_divergent_branches(kernel)
    issues: List[CoopGroupIssue] = []
    group_names: Dict[str, str] = {}  # var_name -> group_type
    groups_found = 0
    sync_points = 0

    # find group declarations
    for m in _CG_GROUP_RE.finditer(kernel):
        gtype = m.group(1)
        gname = m.group(2)
        group_names[gname] = gtype
        groups_found += 1
        lineno = _line_number_in_kernel(kernel, m.start())

        # validate tiled_partition size
        tile_m = _CG_TILE_RE.search(gtype)
        if tile_m:
            size = int(tile_m.group(1))
            if size not in (1, 2, 4, 8, 16, 32):
                issues.append(CoopGroupIssue(
                    lineno, SyncIssueType.MASK_MISMATCH,
                    f"tiled_partition<{size}> size must be a power of 2 and <= 32.",
                    "Use tiled_partition<1>, <2>, <4>, <8>, <16>, or <32>.",
                    gtype,
                ))

    # count this_thread_block() / this_grid() as implicit group usage
    for m in _CG_THIS_RE.finditer(kernel):
        groups_found += 1

    # find .sync() calls
    for m in _CG_SYNC_RE.finditer(kernel):
        sync_points += 1
        var = m.group(1)
        lineno = _line_number_in_kernel(kernel, m.start())
        in_div = _is_inside_divergent(kernel, m.start(), divergent)
        gtype = group_names.get(var, "unknown")

        if in_div:
            issues.append(CoopGroupIssue(
                lineno, SyncIssueType.DIVERGENT_WARP,
                f"Cooperative group sync on '{var}' inside divergent branch.",
                "Move sync outside divergent code to avoid deadlock.",
                gtype,
            ))
        if gtype == "unknown" and var not in group_names:
            issues.append(CoopGroupIssue(
                lineno, SyncIssueType.MISSING_SYNC,
                f"sync() called on undeclared group variable '{var}'.",
                "Declare the cooperative group before calling sync().",
                "unknown",
            ))

    # check for grid_group without cooperative launch
    if any("grid_group" in t for t in group_names.values()):
        if "cudaLaunchCooperativeKernel" not in kernel:
            issues.append(CoopGroupIssue(
                0, SyncIssueType.MISSING_SYNC,
                "grid_group used but no cooperative kernel launch detected.",
                "Launch with cudaLaunchCooperativeKernel for grid-wide sync.",
                "cooperative_groups::grid_group",
            ))

    safe = len(issues) == 0
    return CoopGroupResult(kernel_name, groups_found, issues, sync_points, safe)


# ---------------------------------------------------------------------------
# Tensor core verification
# ---------------------------------------------------------------------------

_WMMA_LOAD_RE = re.compile(r'wmma::load_matrix_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)')
_WMMA_STORE_RE = re.compile(r'wmma::store_matrix_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^)]+)\)')
_WMMA_MMA_RE = re.compile(r'wmma::mma_sync\s*\(\s*([^,]+),\s*([^,]+),\s*([^,]+),\s*([^)]+)\)')
_WMMA_FRAG_RE = re.compile(
    r'wmma::fragment\s*<\s*wmma::(\w+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\w+)\s*>\s+(\w+)'
)
_SYNCTHREADS_RE = re.compile(r'__syncthreads\s*\(\s*\)')


def verify_tensor_core_sync(kernel: str) -> TensorCoreResult:
    """Parse and verify tensor core (WMMA) operations in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    divergent = _find_divergent_branches(kernel)
    issues: List[TensorCoreIssue] = []
    mma_ops = 0

    fragments: Dict[str, Tuple[int, str]] = {}  # name -> (line, role)
    load_lines: List[int] = []
    mma_lines: List[int] = []
    store_lines: List[int] = []
    sync_lines: List[int] = []

    # parse fragments
    for m in _WMMA_FRAG_RE.finditer(kernel):
        role = m.group(1)  # matrix_a, matrix_b, accumulator
        frag_name = m.group(6)
        lineno = _line_number_in_kernel(kernel, m.start())
        fragments[frag_name] = (lineno, role)

    # parse loads
    for m in _WMMA_LOAD_RE.finditer(kernel):
        lineno = _line_number_in_kernel(kernel, m.start())
        load_lines.append(lineno)
        in_div = _is_inside_divergent(kernel, m.start(), divergent)
        if in_div:
            issues.append(TensorCoreIssue(
                lineno, SyncIssueType.DIVERGENT_WARP,
                "wmma::load_matrix_sync inside divergent branch — all warp threads must participate.",
                "Move WMMA load outside divergent code.",
            ))

    # parse stores
    for m in _WMMA_STORE_RE.finditer(kernel):
        lineno = _line_number_in_kernel(kernel, m.start())
        store_lines.append(lineno)
        in_div = _is_inside_divergent(kernel, m.start(), divergent)
        if in_div:
            issues.append(TensorCoreIssue(
                lineno, SyncIssueType.DIVERGENT_WARP,
                "wmma::store_matrix_sync inside divergent branch.",
                "Move WMMA store outside divergent code.",
            ))

    # parse MMA ops
    for m in _WMMA_MMA_RE.finditer(kernel):
        mma_ops += 1
        lineno = _line_number_in_kernel(kernel, m.start())
        mma_lines.append(lineno)
        in_div = _is_inside_divergent(kernel, m.start(), divergent)
        if in_div:
            issues.append(TensorCoreIssue(
                lineno, SyncIssueType.DIVERGENT_WARP,
                "wmma::mma_sync inside divergent branch.",
                "Move MMA outside divergent code.",
            ))

    # collect __syncthreads positions
    for m in _SYNCTHREADS_RE.finditer(kernel):
        sync_lines.append(_line_number_in_kernel(kernel, m.start()))

    # check: shared-memory loads should be preceded by __syncthreads if
    # shared memory is written earlier
    shared_write_re = re.compile(r'\b__shared__\b')
    has_shared = bool(shared_write_re.search(kernel))
    if has_shared and load_lines:
        for ld in load_lines:
            preceding_syncs = [s for s in sync_lines if s < ld]
            if not preceding_syncs:
                issues.append(TensorCoreIssue(
                    ld, SyncIssueType.RACE_CONDITION,
                    "wmma::load_matrix_sync reads shared memory without preceding __syncthreads.",
                    "Add __syncthreads() before loading shared memory into WMMA fragments.",
                ))

    # check: MMA followed by store without sync
    for mma_l in mma_lines:
        later_stores = [s for s in store_lines if s > mma_l]
        if later_stores:
            first_store = later_stores[0]
            interleaved_syncs = [s for s in sync_lines if mma_l < s < first_store]
            # between MMA and store a sync is not strictly required since
            # mma_sync is already warp-synchronous, but if shared memory is
            # involved we flag it
            if has_shared and not interleaved_syncs:
                issues.append(TensorCoreIssue(
                    first_store, SyncIssueType.MISSING_SYNC,
                    "No __syncthreads between MMA and shared-memory store.",
                    "Add __syncthreads() between wmma::mma_sync and shared-memory writes.",
                ))

    sync_correct = len(issues) == 0
    return TensorCoreResult(kernel_name, mma_ops, issues, sync_correct)


# ---------------------------------------------------------------------------
# Warp divergence analysis
# ---------------------------------------------------------------------------

_IF_RE = re.compile(r'\bif\s*\(([^)]+)\)')
_ELSE_RE = re.compile(r'\belse\b')
_FOR_RE = re.compile(r'\bfor\s*\(([^)]*);([^;]*);([^)]*)\)')
_WHILE_RE = re.compile(r'\bwhile\s*\(([^)]+)\)')
_SWITCH_RE = re.compile(r'\bswitch\s*\(([^)]+)\)')
_RETURN_RE = re.compile(r'\breturn\b')
_THREAD_DEP_RE = re.compile(r'\b(threadIdx\.\w|laneid|lane_id|tid)\b', re.IGNORECASE)


def _count_instructions(kernel: str) -> int:
    """Rough instruction count by counting semicolons outside comments."""
    no_comments = re.sub(r'//[^\n]*', '', kernel)
    no_comments = re.sub(r'/\*.*?\*/', '', no_comments, flags=re.DOTALL)
    return no_comments.count(';')


def warp_divergence_analysis(kernel: str) -> DivergenceReport:
    """Analyze warp divergence patterns in *kernel*."""
    kernel_name = _extract_kernel_name(kernel)
    points: List[DivergencePoint] = []
    total_insns = _count_instructions(kernel)
    serialized = 0

    # --- if/else divergence ---
    for m in _IF_RE.finditer(kernel):
        cond = m.group(1).strip()
        if not _THREAD_DEP_RE.search(cond):
            continue
        lineno = _line_number_in_kernel(kernel, m.start())
        reconv = _find_reconvergence(kernel, lineno)
        # estimate affected threads
        if "%" in cond:
            affected = "threads with specific modular condition"
        elif "<" in cond or ">" in cond:
            affected = "subset of threads based on comparison"
        else:
            affected = "variable subset of threads"

        span = (reconv - lineno) if reconv else 5
        body_insns = max(1, span)
        serialized += body_insns

        points.append(DivergencePoint(
            lineno, DivergenceKind.BRANCH_DIVERGENCE, cond, reconv, affected,
        ))

    # --- loop divergence ---
    for m in _FOR_RE.finditer(kernel):
        cond = m.group(2).strip()
        if not _THREAD_DEP_RE.search(cond):
            continue
        lineno = _line_number_in_kernel(kernel, m.start())
        reconv = _find_reconvergence(kernel, lineno)
        serialized += max(1, (reconv - lineno) if reconv else 5)
        points.append(DivergencePoint(
            lineno, DivergenceKind.LOOP_DIVERGENCE, cond, reconv,
            "threads with different iteration counts",
        ))

    for m in _WHILE_RE.finditer(kernel):
        cond = m.group(1).strip()
        if not _THREAD_DEP_RE.search(cond):
            continue
        lineno = _line_number_in_kernel(kernel, m.start())
        reconv = _find_reconvergence(kernel, lineno)
        serialized += max(1, (reconv - lineno) if reconv else 5)
        points.append(DivergencePoint(
            lineno, DivergenceKind.LOOP_DIVERGENCE, cond, reconv,
            "threads with different loop exit conditions",
        ))

    # --- switch divergence ---
    for m in _SWITCH_RE.finditer(kernel):
        cond = m.group(1).strip()
        if not _THREAD_DEP_RE.search(cond):
            continue
        lineno = _line_number_in_kernel(kernel, m.start())
        reconv = _find_reconvergence(kernel, lineno)
        serialized += max(1, (reconv - lineno) if reconv else 10)
        points.append(DivergencePoint(
            lineno, DivergenceKind.SWITCH_DIVERGENCE, cond, reconv,
            "threads selecting different switch cases",
        ))

    # --- return divergence ---
    for m in _RETURN_RE.finditer(kernel):
        lineno = _line_number_in_kernel(kernel, m.start())
        # check if this return is inside a threadIdx-dependent branch
        if any(bl <= lineno <= (rv or bl + 30)
               for bl, _ in _find_divergent_branches(kernel)
               for rv in [_find_reconvergence(kernel, bl)]):
            serialized += 1
            points.append(DivergencePoint(
                lineno, DivergenceKind.RETURN_DIVERGENCE,
                "early return in divergent path", None,
                "threads that take the early return path",
            ))

    # --- indirect call divergence ---
    fptr_re = re.compile(r'\(\*\s*(\w+)\s*\)\s*\(')
    for m in fptr_re.finditer(kernel):
        lineno = _line_number_in_kernel(kernel, m.start())
        serialized += 2
        points.append(DivergencePoint(
            lineno, DivergenceKind.INDIRECT_CALL,
            f"indirect call via {m.group(1)}", None,
            "threads calling different function targets",
        ))

    # warp efficiency: fraction of non-serialized instructions
    if total_insns > 0:
        active_ratio = max(0.0, 1.0 - (serialized / total_insns))
        efficiency = min(1.0, active_ratio)
    else:
        efficiency = 1.0

    return DivergenceReport(
        kernel_name, points, efficiency, serialized, total_insns,
    )
