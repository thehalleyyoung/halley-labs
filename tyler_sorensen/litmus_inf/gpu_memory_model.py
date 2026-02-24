"""
GPU memory model tools for CUDA, Vulkan, and OpenCL.

Validates atomics, barriers, memory consistency, and detects GPU-specific
data races across GPU programming models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
import re


class IssueSeverity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


class ScopeLevel(Enum):
    THREAD = "thread"
    WARP = "warp"
    BLOCK = "block"
    DEVICE = "device"
    SYSTEM = "system"


@dataclass
class AtomicIssue:
    line: int
    description: str
    severity: IssueSeverity = IssueSeverity.ERROR
    suggestion: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.value}] line {self.line}: {self.description}"


@dataclass
class BarrierIssue:
    line: int
    description: str
    severity: IssueSeverity = IssueSeverity.ERROR
    suggestion: str = ""

    def __str__(self) -> str:
        return f"[{self.severity.value}] line {self.line}: {self.description}"


@dataclass
class GPURace:
    line1: int
    line2: int
    variable: str
    scope: ScopeLevel
    description: str = ""

    def __str__(self) -> str:
        return (
            f"GPU race on '{self.variable}' (scope: {self.scope.value}): "
            f"lines {self.line1} and {self.line2} — {self.description}"
        )


@dataclass
class ConsistencyReport:
    consistent: bool
    issues: List[str] = field(default_factory=list)
    memory_model: str = ""
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "consistent" if self.consistent else "INCONSISTENT"
        lines = [f"OpenCL Consistency: {status} (model: {self.memory_model})"]
        for issue in self.issues:
            lines.append(f"  - {issue}")
        for rec in self.recommendations:
            lines.append(f"  Recommendation: {rec}")
        return "\n".join(lines)


# ---------- CUDA analysis ----------

_CUDA_ATOMIC_OPS = [
    r"atomicAdd\s*\(",
    r"atomicSub\s*\(",
    r"atomicExch\s*\(",
    r"atomicMin\s*\(",
    r"atomicMax\s*\(",
    r"atomicInc\s*\(",
    r"atomicDec\s*\(",
    r"atomicCAS\s*\(",
    r"atomicAnd\s*\(",
    r"atomicOr\s*\(",
    r"atomicXor\s*\(",
    r"cuda::atomic",
    r"cuda::std::atomic",
]

_CUDA_FENCES = [
    r"__threadfence_block\s*\(",
    r"__threadfence\s*\(",
    r"__threadfence_system\s*\(",
    r"cuda::atomic_thread_fence",
]

_CUDA_SYNCTHREADS = [
    r"__syncthreads\s*\(",
    r"__syncwarp\s*\(",
    r"cooperative_groups.*sync\(",
]

_CUDA_SHARED = r"__shared__\s+\w+\s+(\w+)"
_CUDA_GLOBAL_STORE = r"(\w+)\s*\[\s*\w+\s*\]\s*="


def validate_cuda_atomics(kernel: str) -> List[AtomicIssue]:
    """Validate CUDA atomic operations for correctness.

    Checks for scope mismatches, missing fences around non-atomic accesses
    adjacent to atomics, and deprecated patterns.

    Args:
        kernel: CUDA kernel source code.

    Returns:
        List of AtomicIssue objects.
    """
    issues: List[AtomicIssue] = []
    lines = kernel.splitlines()

    atomic_lines: List[Tuple[int, str]] = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        for pat in _CUDA_ATOMIC_OPS:
            if re.search(pat, stripped):
                atomic_lines.append((i, stripped))
                break

    fence_lines: set = set()
    for i, line in enumerate(lines, 1):
        for pat in _CUDA_FENCES:
            if re.search(pat, line):
                fence_lines.add(i)

    sync_lines: set = set()
    for i, line in enumerate(lines, 1):
        for pat in _CUDA_SYNCTHREADS:
            if re.search(pat, line):
                sync_lines.add(i)

    for line_no, code in atomic_lines:
        if re.search(r"atomicCAS", code):
            if not any(abs(line_no - f) <= 3 for f in fence_lines):
                nearby_sync = any(abs(line_no - s) <= 5 for s in sync_lines)
                if not nearby_sync:
                    issues.append(AtomicIssue(
                        line=line_no,
                        description="atomicCAS without nearby fence or __syncthreads; "
                                    "non-atomic accesses to same location may be reordered",
                        severity=IssueSeverity.WARNING,
                        suggestion="Add __threadfence() after atomicCAS if publishing data to other blocks",
                    ))

        if re.search(r"atomicAdd\s*\(\s*&?\w+\s*,\s*1\s*\)", code):
            issues.append(AtomicIssue(
                line=line_no,
                description="atomicAdd used as counter; consider atomicInc for bounded counters",
                severity=IssueSeverity.INFO,
                suggestion="atomicInc wraps at a limit and may be more appropriate",
            ))

    for i, line in enumerate(lines, 1):
        m = re.search(_CUDA_SHARED, line)
        if m:
            var = m.group(1)
            has_sync_before_use = False
            for j in range(i + 1, min(len(lines) + 1, i + 15)):
                if j - 1 < len(lines) and re.search(rf"\b{re.escape(var)}\b", lines[j - 1]):
                    if any(s in range(i + 1, j) for s in sync_lines):
                        has_sync_before_use = True
                    break

            store_after = False
            read_after = False
            for j in range(i + 1, min(len(lines) + 1, i + 20)):
                l = lines[j - 1] if j - 1 < len(lines) else ""
                if re.search(rf"\b{re.escape(var)}\b.*=", l):
                    store_after = True
                if re.search(rf"=.*\b{re.escape(var)}\b", l) and store_after:
                    read_after = True
                    if not any(s in range(i + 1, j) for s in sync_lines):
                        issues.append(AtomicIssue(
                            line=j,
                            description=f"Read of shared variable '{var}' after store "
                                        f"without __syncthreads barrier",
                            severity=IssueSeverity.ERROR,
                            suggestion="Add __syncthreads() between shared memory write and read phases",
                        ))
                    break

    return issues


# ---------- Vulkan analysis ----------

_VK_BARRIER = r"vkCmdPipelineBarrier\w*\s*\("
_VK_MEMORY_BARRIER = r"VkMemoryBarrier"
_VK_BUFFER_BARRIER = r"VkBufferMemoryBarrier"
_VK_IMAGE_BARRIER = r"VkImageMemoryBarrier"
_VK_SUBPASS_DEP = r"VkSubpassDependency"

_GLSL_BARRIER = r"\bbarrier\s*\(\s*\)"
_GLSL_MEMBARRIER = [
    r"memoryBarrier\s*\(",
    r"memoryBarrierShared\s*\(",
    r"memoryBarrierBuffer\s*\(",
    r"memoryBarrierImage\s*\(",
    r"groupMemoryBarrier\s*\(",
]

_GLSL_SHARED = r"\bshared\s+\w+\s+(\w+)"
_GLSL_STORAGE = r"\bbuffer\s+\w+"


def validate_vulkan_barriers(shader: str) -> List[BarrierIssue]:
    """Validate Vulkan/GLSL barrier and memory barrier usage.

    Args:
        shader: GLSL compute shader or Vulkan C++ host code.

    Returns:
        List of BarrierIssue objects.
    """
    issues: List[BarrierIssue] = []
    lines = shader.splitlines()
    is_glsl = any(re.search(r"#version|layout\s*\(|gl_", l) for l in lines)

    if is_glsl:
        barrier_lines: List[int] = []
        mem_barrier_lines: List[int] = []
        shared_vars: List[Tuple[int, str]] = []

        for i, line in enumerate(lines, 1):
            if re.search(_GLSL_BARRIER, line):
                barrier_lines.append(i)
            for pat in _GLSL_MEMBARRIER:
                if re.search(pat, line):
                    mem_barrier_lines.append(i)
                    break
            m = re.search(_GLSL_SHARED, line)
            if m:
                shared_vars.append((i, m.group(1)))

        for decl_line, var in shared_vars:
            writes: List[int] = []
            reads: List[int] = []
            for i, line in enumerate(lines, 1):
                if i == decl_line:
                    continue
                if re.search(rf"\b{re.escape(var)}\b", line):
                    if re.search(rf"\b{re.escape(var)}\b\s*(\[.*\])?\s*=", line):
                        writes.append(i)
                    else:
                        reads.append(i)

            for w in writes:
                for r in reads:
                    if r > w:
                        has_barrier = any(w < b < r for b in barrier_lines)
                        has_mem_barrier = any(w < b < r for b in mem_barrier_lines)
                        if not has_barrier and not has_mem_barrier:
                            issues.append(BarrierIssue(
                                line=r,
                                description=f"Read of shared '{var}' at line {r} after "
                                            f"write at line {w} without barrier()",
                                severity=IssueSeverity.ERROR,
                                suggestion="Add barrier() between shared memory write and read phases",
                            ))
                        break

        if not barrier_lines and shared_vars:
            issues.append(BarrierIssue(
                line=1,
                description="Shared variables used but no barrier() calls found",
                severity=IssueSeverity.WARNING,
                suggestion="Add barrier() to synchronize shared memory accesses across invocations",
            ))

    else:
        # Vulkan host-side barrier analysis
        has_barrier = bool(re.search(_VK_BARRIER, shader))
        has_dispatch = bool(re.search(r"vkCmdDispatch", shader))
        has_draw = bool(re.search(r"vkCmdDraw", shader))

        if (has_dispatch or has_draw) and not has_barrier:
            issues.append(BarrierIssue(
                line=1,
                description="Dispatch/Draw commands without pipeline barriers",
                severity=IssueSeverity.WARNING,
                suggestion="Add vkCmdPipelineBarrier between dependent dispatches/draws",
            ))

        for i, line in enumerate(lines, 1):
            if re.search(_VK_MEMORY_BARRIER, line):
                window = "\n".join(lines[max(0, i - 5):min(len(lines), i + 5)])
                if "srcAccessMask" in window and "0" in window:
                    issues.append(BarrierIssue(
                        line=i,
                        description="Memory barrier with srcAccessMask = 0 provides no synchronization",
                        severity=IssueSeverity.WARNING,
                        suggestion="Set srcAccessMask to the access type being synchronized",
                    ))

    return issues


# ---------- OpenCL analysis ----------

_OCL_BARRIER = r"barrier\s*\(\s*(CLK_LOCAL_MEM_FENCE|CLK_GLOBAL_MEM_FENCE)"
_OCL_ATOMIC = [
    r"atomic_add\s*\(",
    r"atomic_sub\s*\(",
    r"atomic_xchg\s*\(",
    r"atomic_cmpxchg\s*\(",
    r"atomic_min\s*\(",
    r"atomic_max\s*\(",
    r"atom_add\s*\(",
    r"atomic_work_item_fence\s*\(",
]
_OCL_LOCAL = r"__local\s+\w+\s+(\w+)"
_OCL_GLOBAL = r"__global\s+\w+\s*\*\s*(\w+)"


def opencl_memory_consistency(kernel: str) -> ConsistencyReport:
    """Analyse OpenCL kernel for memory consistency issues.

    Checks barrier scopes, fence usage, and local/global memory access patterns.

    Args:
        kernel: OpenCL kernel source code.

    Returns:
        ConsistencyReport.
    """
    issues: List[str] = []
    recommendations: List[str] = []
    lines = kernel.splitlines()

    local_vars: List[Tuple[int, str]] = []
    global_vars: List[Tuple[int, str]] = []
    barrier_lines: List[Tuple[int, str]] = []
    atomic_lines: List[int] = []

    for i, line in enumerate(lines, 1):
        m = re.search(_OCL_LOCAL, line)
        if m:
            local_vars.append((i, m.group(1)))
        m = re.search(_OCL_GLOBAL, line)
        if m:
            global_vars.append((i, m.group(1)))
        m = re.search(_OCL_BARRIER, line)
        if m:
            barrier_lines.append((i, m.group(1)))
        for pat in _OCL_ATOMIC:
            if re.search(pat, line):
                atomic_lines.append(i)
                break

    for decl_line, var in local_vars:
        accesses = [(i, l) for i, l in enumerate(lines, 1)
                    if re.search(rf"\b{re.escape(var)}\b", l) and i != decl_line]
        writes = [i for i, l in accesses if re.search(rf"\b{re.escape(var)}\b\s*(\[.*\])?\s*=", l)]
        reads = [i for i, l in accesses if i not in writes]

        for w in writes:
            for r in reads:
                if r > w:
                    has_local_fence = any(
                        w < bl < r and "CLK_LOCAL_MEM_FENCE" in scope
                        for bl, scope in barrier_lines
                    )
                    if not has_local_fence:
                        issues.append(
                            f"Local var '{var}': write at line {w}, read at line {r} "
                            f"without CLK_LOCAL_MEM_FENCE barrier"
                        )
                    break

    for decl_line, var in global_vars:
        accesses = [(i, l) for i, l in enumerate(lines, 1)
                    if re.search(rf"\b{re.escape(var)}\b", l) and i != decl_line]
        writes = [i for i, l in accesses if re.search(rf"\b{re.escape(var)}\b\s*(\[.*\])?\s*=", l)]
        if writes and not any("CLK_GLOBAL_MEM_FENCE" in scope for _, scope in barrier_lines):
            if not atomic_lines:
                issues.append(
                    f"Global var '{var}' written without CLK_GLOBAL_MEM_FENCE barrier or atomics"
                )

    if local_vars and not barrier_lines:
        recommendations.append("Add barrier(CLK_LOCAL_MEM_FENCE) between local memory write and read phases")
    if global_vars and not any("CLK_GLOBAL_MEM_FENCE" in s for _, s in barrier_lines):
        recommendations.append("Consider barrier(CLK_GLOBAL_MEM_FENCE) or atomics for global memory synchronization")

    if re.search(r"get_local_id|get_group_id", kernel):
        if not barrier_lines and (local_vars or global_vars):
            issues.append("Work-items access shared memory but no barriers found")

    return ConsistencyReport(
        consistent=len(issues) == 0,
        issues=issues,
        memory_model="OpenCL 2.0+ relaxed",
        recommendations=recommendations,
    )


def suggest_gpu_fences(kernel: str) -> str:
    """Add minimum necessary fences to a GPU kernel.

    Analyses shared memory access patterns and inserts fences/barriers
    at required synchronization points.

    Args:
        kernel: CUDA or OpenCL kernel source.

    Returns:
        Modified kernel source with fences inserted.
    """
    lines = kernel.splitlines()
    is_cuda = any("__shared__" in l or "__global__" in l for l in lines)
    is_opencl = any("__kernel" in l or "__local" in l for l in lines)

    if is_cuda:
        shared_pat = _CUDA_SHARED
        barrier_call = "    __syncthreads();"
        fence_call = "    __threadfence();"
    elif is_opencl:
        shared_pat = _OCL_LOCAL
        barrier_call = "    barrier(CLK_LOCAL_MEM_FENCE);"
        fence_call = "    barrier(CLK_GLOBAL_MEM_FENCE);"
    else:
        barrier_call = "    barrier();"
        fence_call = "    memoryBarrier();"
        shared_pat = _GLSL_SHARED

    shared_vars: set = set()
    for line in lines:
        m = re.search(shared_pat, line)
        if m:
            shared_vars.add(m.group(1))

    if not shared_vars:
        return kernel

    insertion_points: set = set()
    last_write: Dict[str, int] = {}

    for i, line in enumerate(lines):
        for var in shared_vars:
            if re.search(rf"\b{re.escape(var)}\b", line):
                is_write = bool(re.search(
                    rf"\b{re.escape(var)}\b\s*(\[.*\])?\s*=", line
                ))
                if is_write:
                    last_write[var] = i
                elif var in last_write:
                    write_line = last_write[var]
                    has_existing = any(
                        re.search(r"__syncthreads|barrier\(|__threadfence", lines[j])
                        for j in range(write_line + 1, i)
                    )
                    if not has_existing:
                        insertion_points.add(write_line + 1)

    result_lines = []
    for i, line in enumerate(lines):
        if i in insertion_points:
            result_lines.append(barrier_call)
        result_lines.append(line)

    return "\n".join(result_lines)


def gpu_race_analysis(kernel: str) -> List[GPURace]:
    """Detect data races in GPU kernel code.

    Identifies unsynchronized accesses to shared/local memory across
    threads within a block/workgroup.

    Args:
        kernel: GPU kernel source (CUDA, OpenCL, or GLSL compute).

    Returns:
        List of GPURace objects.
    """
    races: List[GPURace] = []
    lines = kernel.splitlines()

    is_cuda = any("__shared__" in l for l in lines)
    is_opencl = any("__local" in l for l in lines)

    if is_cuda:
        shared_pat = _CUDA_SHARED
        barrier_pats = _CUDA_SYNCTHREADS
        scope = ScopeLevel.BLOCK
    elif is_opencl:
        shared_pat = _OCL_LOCAL
        barrier_pats = [_OCL_BARRIER]
        scope = ScopeLevel.BLOCK
    else:
        shared_pat = _GLSL_SHARED
        barrier_pats = [_GLSL_BARRIER]
        scope = ScopeLevel.BLOCK

    shared_vars: List[Tuple[int, str]] = []
    barrier_lines: List[int] = []

    for i, line in enumerate(lines, 1):
        m = re.search(shared_pat, line)
        if m:
            shared_vars.append((i, m.group(1)))
        for pat in barrier_pats:
            if re.search(pat, line):
                barrier_lines.append(i)

    for _, var in shared_vars:
        writes: List[int] = []
        reads: List[int] = []
        for i, line in enumerate(lines, 1):
            if re.search(rf"\b{re.escape(var)}\b", line):
                if re.search(rf"\b{re.escape(var)}\b\s*(\[.*\])?\s*=", line):
                    writes.append(i)
                elif not re.search(rf"(shared|__shared__|__local)\s+\w+\s+{re.escape(var)}", line):
                    reads.append(i)

        for w in writes:
            for r in reads:
                if r > w:
                    barrier_between = any(w < b < r for b in barrier_lines)
                    if not barrier_between:
                        races.append(GPURace(
                            line1=w, line2=r, variable=var, scope=scope,
                            description=f"Write at line {w} and read at line {r} "
                                        f"without barrier synchronization",
                        ))
            for w2 in writes:
                if w2 > w:
                    barrier_between = any(w < b < w2 for b in barrier_lines)
                    if not barrier_between:
                        races.append(GPURace(
                            line1=w, line2=w2, variable=var, scope=scope,
                            description=f"Concurrent writes at lines {w} and {w2} "
                                        f"without barrier synchronization",
                        ))

    seen = set()
    deduped = []
    for r in races:
        key = (r.line1, r.line2, r.variable)
        if key not in seen:
            seen.add(key)
            deduped.append(r)

    return deduped


# =============================================================================
# Extended GPU memory model: formal model checking, GPU litmus tests,
# scope-based memory model verification, and GPU race detection.
# =============================================================================

import numpy as np
import itertools
from typing import Set, Any, Callable


class GPUScopeLevel(Enum):
    THREAD = "thread"
    WARP = "warp"
    CTA = "cta"
    GPU = "gpu"
    SYSTEM = "system"


class GPUMemorySpace(Enum):
    GLOBAL = "global"
    SHARED = "shared"
    LOCAL = "local"
    CONSTANT = "constant"
    TEXTURE = "texture"


class GPUFenceType(Enum):
    THREADFENCE_BLOCK = "threadfence_block"
    THREADFENCE = "threadfence"
    THREADFENCE_SYSTEM = "threadfence_system"
    SYNCTHREADS = "syncthreads"
    SYNCWARP = "syncwarp"


class GPUAtomicOp(Enum):
    ATOMIC_ADD = "atomicAdd"
    ATOMIC_CAS = "atomicCAS"
    ATOMIC_EXCH = "atomicExch"
    ATOMIC_MIN = "atomicMin"
    ATOMIC_MAX = "atomicMax"
    ATOMIC_AND = "atomicAnd"
    ATOMIC_OR = "atomicOr"
    ATOMIC_XOR = "atomicXor"


@dataclass
class GPUEvent:
    eid: int
    thread_id: int
    warp_id: int
    cta_id: int
    gpu_id: int
    event_type: str  # "R", "W", "F", "RMW"
    variable: str = ""
    value: int = 0
    memory_space: GPUMemorySpace = GPUMemorySpace.GLOBAL
    scope: GPUScopeLevel = GPUScopeLevel.SYSTEM
    fence_type: Optional[GPUFenceType] = None
    atomic_op: Optional[GPUAtomicOp] = None

    def __hash__(self):
        return hash(self.eid)

    def __eq__(self, other):
        return isinstance(other, GPUEvent) and self.eid == other.eid

    def __repr__(self):
        return (f"GPUEvent({self.eid}: T{self.thread_id}/W{self.warp_id}"
                f"/CTA{self.cta_id} {self.event_type} {self.variable}={self.value}"
                f" @{self.scope.value})")


@dataclass
class GPUExecution:
    events: List[GPUEvent] = field(default_factory=list)
    po: Set[Tuple[int, int]] = field(default_factory=set)
    rf: Set[Tuple[int, int]] = field(default_factory=set)
    co: Set[Tuple[int, int]] = field(default_factory=set)
    initial_state: Dict[str, int] = field(default_factory=dict)

    def add_event(self, event: GPUEvent):
        self.events.append(event)

    def get_outcome(self) -> Dict[str, int]:
        final = dict(self.initial_state)
        writes_by_var: Dict[str, List[GPUEvent]] = {}
        for e in self.events:
            if e.event_type == "W":
                writes_by_var.setdefault(e.variable, []).append(e)
        for var, writes in writes_by_var.items():
            if writes:
                final[var] = writes[-1].value
        return final


@dataclass
class GPUScopeTree:
    thread_to_warp: Dict[int, int] = field(default_factory=dict)
    thread_to_cta: Dict[int, int] = field(default_factory=dict)
    thread_to_gpu: Dict[int, int] = field(default_factory=dict)
    warp_size: int = 32
    cta_size: int = 256

    def same_scope(self, tid1: int, tid2: int, scope: GPUScopeLevel) -> bool:
        if scope == GPUScopeLevel.THREAD:
            return tid1 == tid2
        elif scope == GPUScopeLevel.WARP:
            return self.thread_to_warp.get(tid1, -1) == self.thread_to_warp.get(tid2, -2)
        elif scope == GPUScopeLevel.CTA:
            return self.thread_to_cta.get(tid1, -1) == self.thread_to_cta.get(tid2, -2)
        elif scope == GPUScopeLevel.GPU:
            return self.thread_to_gpu.get(tid1, -1) == self.thread_to_gpu.get(tid2, -2)
        return True  # SYSTEM scope: all threads


class PTXMemoryModel:
    """PTX memory model with scopes: CTA, GPU, System."""

    def __init__(self, scope_tree: GPUScopeTree = None):
        self.scope_tree = scope_tree or GPUScopeTree()
        self._scope_order = [
            GPUScopeLevel.THREAD, GPUScopeLevel.WARP,
            GPUScopeLevel.CTA, GPUScopeLevel.GPU, GPUScopeLevel.SYSTEM
        ]

    def check(self, execution: GPUExecution) -> Tuple[bool, List[str]]:
        violations = []

        if not self._check_coherence(execution):
            violations.append("Coherence violation")

        if not self._check_scope_visibility(execution):
            violations.append("Scope visibility violation")

        if not self._check_fence_ordering(execution):
            violations.append("Fence ordering violation")

        return len(violations) == 0, violations

    def _check_coherence(self, execution: GPUExecution) -> bool:
        # Per-location coherence: co;rf and fr;rf must be acyclic
        variables = set(e.variable for e in execution.events if e.variable)
        for var in variables:
            writes = [e for e in execution.events
                      if e.event_type == "W" and e.variable == var]
            if len(writes) <= 1:
                continue
            # Check co is consistent with rf
            for w1 in writes:
                for w2 in writes:
                    if w1.eid != w2.eid:
                        if ((w1.eid, w2.eid) in execution.co and
                                (w2.eid, w1.eid) in execution.co):
                            return False
        return True

    def _check_scope_visibility(self, execution: GPUExecution) -> bool:
        for src, dst in execution.rf:
            src_ev = next((e for e in execution.events if e.eid == src), None)
            dst_ev = next((e for e in execution.events if e.eid == dst), None)
            if src_ev and dst_ev:
                needed_scope = self._min_scope(src_ev.thread_id, dst_ev.thread_id)
                write_scope_idx = self._scope_order.index(src_ev.scope)
                needed_idx = self._scope_order.index(needed_scope)
                if write_scope_idx < needed_idx:
                    return False
        return True

    def _check_fence_ordering(self, execution: GPUExecution) -> bool:
        fences = [e for e in execution.events if e.event_type == "F"]
        for fence in fences:
            before = [e for e in execution.events
                      if e.thread_id == fence.thread_id and e.eid < fence.eid
                      and e.event_type in ("R", "W")]
            after = [e for e in execution.events
                     if e.thread_id == fence.thread_id and e.eid > fence.eid
                     and e.event_type in ("R", "W")]

            fence_scope = self._fence_scope(fence)

            for b in before:
                for a in after:
                    if b.event_type == "W" and a.event_type == "R":
                        if not self.scope_tree.same_scope(
                                b.thread_id, a.thread_id, fence_scope):
                            pass
        return True

    def _min_scope(self, tid1: int, tid2: int) -> GPUScopeLevel:
        for scope in self._scope_order:
            if self.scope_tree.same_scope(tid1, tid2, scope):
                return scope
        return GPUScopeLevel.SYSTEM

    def _fence_scope(self, fence: GPUEvent) -> GPUScopeLevel:
        if fence.fence_type == GPUFenceType.THREADFENCE_BLOCK:
            return GPUScopeLevel.CTA
        elif fence.fence_type == GPUFenceType.THREADFENCE:
            return GPUScopeLevel.GPU
        elif fence.fence_type == GPUFenceType.THREADFENCE_SYSTEM:
            return GPUScopeLevel.SYSTEM
        elif fence.fence_type == GPUFenceType.SYNCTHREADS:
            return GPUScopeLevel.CTA
        elif fence.fence_type == GPUFenceType.SYNCWARP:
            return GPUScopeLevel.WARP
        return GPUScopeLevel.GPU


class VulkanMemoryModel:
    """Vulkan memory model: availability and visibility operations."""

    def __init__(self):
        self._availability_ops: Dict[int, GPUScopeLevel] = {}
        self._visibility_ops: Dict[int, GPUScopeLevel] = {}

    def check(self, execution: GPUExecution) -> Tuple[bool, List[str]]:
        violations = []

        for src, dst in execution.rf:
            src_ev = next((e for e in execution.events if e.eid == src), None)
            dst_ev = next((e for e in execution.events if e.eid == dst), None)
            if not src_ev or not dst_ev:
                continue

            if src_ev.thread_id != dst_ev.thread_id:
                if not self._is_available(src_ev, execution):
                    violations.append(
                        f"Write E{src} not made available before read E{dst}")
                if not self._is_visible(dst_ev, execution):
                    violations.append(
                        f"Read E{dst} without visibility operation")

        return len(violations) == 0, violations

    def _is_available(self, write_ev: GPUEvent,
                      execution: GPUExecution) -> bool:
        for e in execution.events:
            if (e.thread_id == write_ev.thread_id and
                    e.eid > write_ev.eid and
                    e.event_type == "F"):
                return True
        return write_ev.scope == GPUScopeLevel.SYSTEM

    def _is_visible(self, read_ev: GPUEvent,
                    execution: GPUExecution) -> bool:
        for e in execution.events:
            if (e.thread_id == read_ev.thread_id and
                    e.eid < read_ev.eid and
                    e.event_type == "F"):
                return True
        return False


class OpenCLMemoryModel:
    """OpenCL memory model: global, local, private memory spaces."""

    def __init__(self):
        self._space_rules = {
            GPUMemorySpace.LOCAL: GPUScopeLevel.CTA,
            GPUMemorySpace.GLOBAL: GPUScopeLevel.SYSTEM,
        }

    def check(self, execution: GPUExecution,
              scope_tree: GPUScopeTree) -> Tuple[bool, List[str]]:
        violations = []

        for src, dst in execution.rf:
            src_ev = next((e for e in execution.events if e.eid == src), None)
            dst_ev = next((e for e in execution.events if e.eid == dst), None)
            if not src_ev or not dst_ev:
                continue

            if src_ev.memory_space == GPUMemorySpace.LOCAL:
                if not scope_tree.same_scope(
                        src_ev.thread_id, dst_ev.thread_id, GPUScopeLevel.CTA):
                    violations.append(
                        f"Local memory access E{src}->E{dst} across work-groups")

            if src_ev.memory_space == GPUMemorySpace.LOCAL:
                if dst_ev.memory_space != GPUMemorySpace.LOCAL:
                    violations.append(
                        f"Mixed memory space access E{src}(local)->E{dst}")

        return len(violations) == 0, violations


class WarpSynchronization:
    """Warp-level synchronization: __syncwarp, warp vote functions."""

    def __init__(self, warp_size: int = 32):
        self.warp_size = warp_size

    def syncwarp(self, execution: GPUExecution, warp_id: int,
                 mask: int = 0xFFFFFFFF) -> Set[Tuple[int, int]]:
        warp_events = [e for e in execution.events if e.warp_id == warp_id]
        sync_fences = [e for e in warp_events
                       if e.event_type == "F" and
                       e.fence_type == GPUFenceType.SYNCWARP]

        ordered_pairs = set()
        for fence in sync_fences:
            before = [e for e in warp_events if e.eid < fence.eid]
            after = [e for e in warp_events if e.eid > fence.eid]
            for b in before:
                for a in after:
                    if b.thread_id != a.thread_id:
                        lane_b = b.thread_id % self.warp_size
                        lane_a = a.thread_id % self.warp_size
                        if (mask >> lane_b) & 1 and (mask >> lane_a) & 1:
                            ordered_pairs.add((b.eid, a.eid))
        return ordered_pairs

    def warp_vote_all(self, predicates: Dict[int, bool], warp_id: int) -> bool:
        warp_preds = {tid: pred for tid, pred in predicates.items()
                      if tid // self.warp_size == warp_id}
        return all(warp_preds.values()) if warp_preds else True

    def warp_vote_any(self, predicates: Dict[int, bool], warp_id: int) -> bool:
        warp_preds = {tid: pred for tid, pred in predicates.items()
                      if tid // self.warp_size == warp_id}
        return any(warp_preds.values()) if warp_preds else False

    def warp_ballot(self, predicates: Dict[int, bool], warp_id: int) -> int:
        result = 0
        for tid, pred in predicates.items():
            if tid // self.warp_size == warp_id and pred:
                lane = tid % self.warp_size
                result |= (1 << lane)
        return result


class GPUAtomicOperations:
    """GPU atomic operations at different scopes."""

    def __init__(self, scope_tree: GPUScopeTree = None):
        self.scope_tree = scope_tree or GPUScopeTree()

    def atomic_add(self, memory: Dict[str, int], var: str, value: int,
                   scope: GPUScopeLevel = GPUScopeLevel.GPU) -> int:
        old = memory.get(var, 0)
        memory[var] = old + value
        return old

    def atomic_cas(self, memory: Dict[str, int], var: str,
                   compare: int, value: int,
                   scope: GPUScopeLevel = GPUScopeLevel.GPU) -> int:
        old = memory.get(var, 0)
        if old == compare:
            memory[var] = value
        return old

    def atomic_exch(self, memory: Dict[str, int], var: str, value: int,
                    scope: GPUScopeLevel = GPUScopeLevel.GPU) -> int:
        old = memory.get(var, 0)
        memory[var] = value
        return old

    def atomic_min(self, memory: Dict[str, int], var: str, value: int) -> int:
        old = memory.get(var, 0)
        memory[var] = min(old, value)
        return old

    def atomic_max(self, memory: Dict[str, int], var: str, value: int) -> int:
        old = memory.get(var, 0)
        memory[var] = max(old, value)
        return old


class GPUMemoryFences:
    """GPU memory fence types and their ordering guarantees."""

    def __init__(self):
        self.fence_scopes = {
            GPUFenceType.THREADFENCE_BLOCK: GPUScopeLevel.CTA,
            GPUFenceType.THREADFENCE: GPUScopeLevel.GPU,
            GPUFenceType.THREADFENCE_SYSTEM: GPUScopeLevel.SYSTEM,
            GPUFenceType.SYNCTHREADS: GPUScopeLevel.CTA,
            GPUFenceType.SYNCWARP: GPUScopeLevel.WARP,
        }

    def orders(self, fence_type: GPUFenceType, before: GPUEvent,
               after: GPUEvent, scope_tree: GPUScopeTree) -> bool:
        fence_scope = self.fence_scopes.get(fence_type, GPUScopeLevel.GPU)

        if fence_type == GPUFenceType.SYNCTHREADS:
            return scope_tree.same_scope(
                before.thread_id, after.thread_id, GPUScopeLevel.CTA)

        if fence_type == GPUFenceType.SYNCWARP:
            return scope_tree.same_scope(
                before.thread_id, after.thread_id, GPUScopeLevel.WARP)

        scope_idx = [GPUScopeLevel.THREAD, GPUScopeLevel.WARP,
                     GPUScopeLevel.CTA, GPUScopeLevel.GPU, GPUScopeLevel.SYSTEM]
        return scope_idx.index(fence_scope) >= scope_idx.index(
            GPUScopeLevel.CTA if scope_tree.same_scope(
                before.thread_id, after.thread_id, GPUScopeLevel.CTA)
            else GPUScopeLevel.GPU)


class GPULitmusTestBuilder:
    """Build GPU-specific litmus tests with scope annotations."""

    def build_gpu_sb(self, same_cta: bool = True) -> GPUExecution:
        """Store Buffer test for GPU."""
        scope_tree = GPUScopeTree()
        if same_cta:
            scope_tree.thread_to_cta = {0: 0, 1: 0}
            scope_tree.thread_to_warp = {0: 0, 1: 0}
        else:
            scope_tree.thread_to_cta = {0: 0, 1: 1}
            scope_tree.thread_to_warp = {0: 0, 1: 1}

        scope = GPUScopeLevel.CTA if same_cta else GPUScopeLevel.GPU

        exec_ = GPUExecution(initial_state={"x": 0, "y": 0})
        exec_.add_event(GPUEvent(0, 0, 0, 0, 0, "W", "x", 1, scope=scope))
        exec_.add_event(GPUEvent(1, 0, 0, 0, 0, "R", "y", 0, scope=scope))
        exec_.add_event(GPUEvent(2, 1, 0 if same_cta else 1,
                                 0 if same_cta else 1, 0, "W", "y", 1, scope=scope))
        exec_.add_event(GPUEvent(3, 1, 0 if same_cta else 1,
                                 0 if same_cta else 1, 0, "R", "x", 0, scope=scope))
        exec_.po = {(0, 1), (2, 3)}
        return exec_

    def build_gpu_mp(self, fence_type: Optional[GPUFenceType] = None,
                     same_cta: bool = True) -> GPUExecution:
        """Message Passing test for GPU with optional fence."""
        scope = GPUScopeLevel.CTA if same_cta else GPUScopeLevel.GPU

        exec_ = GPUExecution(initial_state={"data": 0, "flag": 0})
        exec_.add_event(GPUEvent(0, 0, 0, 0, 0, "W", "data", 1, scope=scope))

        eid = 1
        if fence_type:
            exec_.add_event(GPUEvent(eid, 0, 0, 0, 0, "F", fence_type=fence_type))
            exec_.po.add((0, eid))
            eid += 1

        exec_.add_event(GPUEvent(eid, 0, 0, 0, 0, "W", "flag", 1, scope=scope))
        exec_.po.add((eid - 1, eid))
        flag_write_eid = eid
        eid += 1

        cta_id = 0 if same_cta else 1
        warp_id = 0 if same_cta else 1

        exec_.add_event(GPUEvent(eid, 1, warp_id, cta_id, 0, "R", "flag", scope=scope))
        flag_read_eid = eid
        eid += 1

        if fence_type:
            exec_.add_event(GPUEvent(eid, 1, warp_id, cta_id, 0, "F",
                                     fence_type=fence_type))
            exec_.po.add((flag_read_eid, eid))
            eid += 1

        exec_.add_event(GPUEvent(eid, 1, warp_id, cta_id, 0, "R", "data", scope=scope))
        exec_.po.add((eid - 1, eid))

        return exec_

    def build_gpu_iriw(self, same_cta: bool = False) -> GPUExecution:
        """IRIW test for GPU with 4 threads."""
        exec_ = GPUExecution(initial_state={"x": 0, "y": 0})
        scope = GPUScopeLevel.GPU

        exec_.add_event(GPUEvent(0, 0, 0, 0, 0, "W", "x", 1, scope=scope))
        exec_.add_event(GPUEvent(1, 1, 0, 0, 0, "W", "y", 1, scope=scope))
        exec_.add_event(GPUEvent(2, 2, 1, 1, 0, "R", "x", scope=scope))
        exec_.add_event(GPUEvent(3, 2, 1, 1, 0, "R", "y", scope=scope))
        exec_.add_event(GPUEvent(4, 3, 1, 1, 0, "R", "y", scope=scope))
        exec_.add_event(GPUEvent(5, 3, 1, 1, 0, "R", "x", scope=scope))

        exec_.po = {(2, 3), (4, 5)}
        return exec_

    def build_gpu_coherence(self) -> GPUExecution:
        """Coherence test: two writes to same variable."""
        exec_ = GPUExecution(initial_state={"x": 0})
        exec_.add_event(GPUEvent(0, 0, 0, 0, 0, "W", "x", 1,
                                 scope=GPUScopeLevel.GPU))
        exec_.add_event(GPUEvent(1, 1, 1, 1, 0, "W", "x", 2,
                                 scope=GPUScopeLevel.GPU))
        exec_.add_event(GPUEvent(2, 2, 0, 0, 0, "R", "x",
                                 scope=GPUScopeLevel.GPU))
        exec_.add_event(GPUEvent(3, 3, 1, 1, 0, "R", "x",
                                 scope=GPUScopeLevel.GPU))
        return exec_

    def build_gpu_scope_test(self) -> GPUExecution:
        """Test scope visibility: CTA-scoped write not visible to other CTA."""
        exec_ = GPUExecution(initial_state={"x": 0})
        # CTA-scoped write
        exec_.add_event(GPUEvent(0, 0, 0, 0, 0, "W", "x", 1,
                                 scope=GPUScopeLevel.CTA))
        # Read from different CTA
        exec_.add_event(GPUEvent(1, 1, 1, 1, 0, "R", "x",
                                 scope=GPUScopeLevel.CTA))
        return exec_


class GPURaceDetector:
    """Detect data races considering GPU scope hierarchy."""

    def __init__(self, scope_tree: GPUScopeTree = None):
        self.scope_tree = scope_tree or GPUScopeTree()

    def detect(self, execution: GPUExecution) -> List[Tuple[GPUEvent, GPUEvent]]:
        races = []
        accesses: Dict[str, List[GPUEvent]] = {}

        for e in execution.events:
            if e.variable and e.event_type in ("R", "W"):
                accesses.setdefault(e.variable, []).append(e)

        for var, var_accesses in accesses.items():
            for i, a1 in enumerate(var_accesses):
                for a2 in var_accesses[i + 1:]:
                    if a1.thread_id == a2.thread_id:
                        continue
                    if a1.event_type == "R" and a2.event_type == "R":
                        continue
                    if self._is_ordered(a1, a2, execution):
                        continue
                    if self._is_scope_protected(a1, a2, execution):
                        continue
                    races.append((a1, a2))
        return races

    def _is_ordered(self, a1: GPUEvent, a2: GPUEvent,
                    execution: GPUExecution) -> bool:
        if (a1.eid, a2.eid) in execution.po or (a2.eid, a1.eid) in execution.po:
            return True
        # Check for fence ordering
        fences = [e for e in execution.events if e.event_type == "F"]
        for f in fences:
            if f.thread_id == a1.thread_id and a1.eid < f.eid:
                if (f.eid, a2.eid) in execution.po:
                    return True
            if f.thread_id == a2.thread_id and a2.eid < f.eid:
                if (f.eid, a1.eid) in execution.po:
                    return True
        return False

    def _is_scope_protected(self, a1: GPUEvent, a2: GPUEvent,
                            execution: GPUExecution) -> bool:
        if a1.memory_space == GPUMemorySpace.LOCAL:
            return True
        if a1.memory_space == GPUMemorySpace.SHARED:
            if not self.scope_tree.same_scope(
                    a1.thread_id, a2.thread_id, GPUScopeLevel.CTA):
                return True
        return False


class GPUMemoryModel:
    """Main GPU memory model checker: combines PTX, Vulkan, OpenCL."""

    def __init__(self, scope_tree: GPUScopeTree = None):
        self.scope_tree = scope_tree or GPUScopeTree()
        self.ptx_model = PTXMemoryModel(self.scope_tree)
        self.vulkan_model = VulkanMemoryModel()
        self.opencl_model = OpenCLMemoryModel()
        self.litmus_builder = GPULitmusTestBuilder()
        self.race_detector = GPURaceDetector(self.scope_tree)
        self.fences = GPUMemoryFences()
        self.atomics = GPUAtomicOperations(self.scope_tree)
        self.warp_sync = WarpSynchronization()

    def check(self, execution: GPUExecution,
              model: str = "PTX") -> Tuple[bool, List[str]]:
        if model == "PTX":
            return self.ptx_model.check(execution)
        elif model == "Vulkan":
            return self.vulkan_model.check(execution)
        elif model == "OpenCL":
            return self.opencl_model.check(execution, self.scope_tree)
        return False, [f"Unknown model: {model}"]

    def check_all_models(self, execution: GPUExecution) -> Dict[str, Tuple[bool, List[str]]]:
        return {
            "PTX": self.ptx_model.check(execution),
            "Vulkan": self.vulkan_model.check(execution),
            "OpenCL": self.opencl_model.check(execution, self.scope_tree),
        }

    def detect_races(self, execution: GPUExecution) -> List[Tuple[GPUEvent, GPUEvent]]:
        return self.race_detector.detect(execution)

    def run_gpu_litmus_suite(self) -> Dict[str, Dict[str, Any]]:
        results = {}

        # SB test (same CTA)
        sb_same = self.litmus_builder.build_gpu_sb(same_cta=True)
        allowed, violations = self.check(sb_same, "PTX")
        results["SB_same_CTA"] = {"allowed": allowed, "violations": violations}

        # SB test (different CTA)
        sb_diff = self.litmus_builder.build_gpu_sb(same_cta=False)
        allowed, violations = self.check(sb_diff, "PTX")
        results["SB_diff_CTA"] = {"allowed": allowed, "violations": violations}

        # MP test (no fence)
        mp_nofence = self.litmus_builder.build_gpu_mp(fence_type=None)
        allowed, violations = self.check(mp_nofence, "PTX")
        results["MP_no_fence"] = {"allowed": allowed, "violations": violations}

        # MP test (with threadfence)
        mp_fence = self.litmus_builder.build_gpu_mp(
            fence_type=GPUFenceType.THREADFENCE)
        allowed, violations = self.check(mp_fence, "PTX")
        results["MP_threadfence"] = {"allowed": allowed, "violations": violations}

        # Scope test
        scope_test = self.litmus_builder.build_gpu_scope_test()
        allowed, violations = self.check(scope_test, "PTX")
        results["scope_visibility"] = {"allowed": allowed, "violations": violations}

        return results
