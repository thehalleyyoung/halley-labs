#!/usr/bin/env python3
"""
cuda_checker.py — CUDA-specific concurrency checking for LITMUS∞.

Validates CUDA kernel patterns for:
  - Thread safety across thread blocks
  - Scope mismatch bugs in cooperative groups
  - Warp-level synchronization correctness
  - Memory ordering verification for shared/global memory

Usage:
    from cuda_checker import validate_cuda_kernel, detect_scope_mismatch
    result = validate_cuda_kernel(code, grid_dim=(4,1,1), block_dim=(32,1,1))
"""

import re
import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum

sys.path.insert(0, os.path.dirname(__file__))


# ── Enums & Constants ───────────────────────────────────────────────

class MemSpace(Enum):
    SHARED = "shared"
    GLOBAL = "global"
    LOCAL = "local"


class SyncScope(Enum):
    WARP = "warp"
    BLOCK = "block"       # CTA
    DEVICE = "device"
    SYSTEM = "system"


class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


WARP_SIZE = 32

# PTX membar scopes and their visibility
PTX_MEMBAR_SCOPE = {
    "membar.cta":  SyncScope.BLOCK,
    "membar.gl":   SyncScope.DEVICE,
    "membar.sys":  SyncScope.SYSTEM,
}

# CUDA __syncthreads variants
CUDA_SYNC_SCOPE = {
    "__syncthreads":          SyncScope.BLOCK,
    "__syncwarp":             SyncScope.WARP,
    "__threadfence_block":    SyncScope.BLOCK,
    "__threadfence":          SyncScope.DEVICE,
    "__threadfence_system":   SyncScope.SYSTEM,
    "cooperative_groups::this_thread_block().sync": SyncScope.BLOCK,
    "cooperative_groups::this_grid().sync":         SyncScope.DEVICE,
}

# Memory fence cost model (relative latency)
FENCE_LATENCY = {
    SyncScope.WARP:   1,
    SyncScope.BLOCK:  5,
    SyncScope.DEVICE: 50,
    SyncScope.SYSTEM: 200,
}


# ── Data Classes ────────────────────────────────────────────────────

@dataclass
class CUDAIssue:
    """A single issue found during CUDA kernel validation."""
    severity: Severity
    category: str          # 'scope_mismatch', 'warp_sync', 'memory_ordering', 'thread_safety'
    line: Optional[int]
    description: str
    fix: str
    affected_threads: str  # e.g., "cross-block", "intra-warp"

    def __repr__(self):
        return f"[{self.severity.value}] {self.category} (line {self.line}): {self.description}"


@dataclass
class ScopeMismatch:
    """A detected scope mismatch bug."""
    barrier_type: str
    barrier_scope: SyncScope
    required_scope: SyncScope
    line: Optional[int]
    pattern: str           # communication pattern name
    description: str
    fix: str

    @property
    def is_critical(self) -> bool:
        return self.barrier_scope.value != self.required_scope.value


@dataclass
class WarpSyncIssue:
    """A warp synchronization correctness issue."""
    issue_type: str        # 'divergent_sync', 'missing_mask', 'partial_warp', 'deprecated'
    line: Optional[int]
    description: str
    fix: str
    warp_mask: Optional[str] = None

    def __repr__(self):
        return f"WarpSync({self.issue_type}, line {self.line}): {self.description}"


@dataclass
class MemoryOrderingIssue:
    """A memory ordering violation in shared or global memory."""
    mem_space: MemSpace
    issue_type: str        # 'racy_access', 'missing_fence', 'wrong_scope', 'volatile_needed'
    line: Optional[int]
    description: str
    fix: str
    store_thread: Optional[str] = None
    load_thread: Optional[str] = None

    def __repr__(self):
        return f"MemOrder({self.mem_space.value}/{self.issue_type}, line {self.line})"


@dataclass
class CUDAValidationResult:
    """Complete result of CUDA kernel validation."""
    kernel_name: str
    grid_dim: Tuple[int, ...]
    block_dim: Tuple[int, ...]
    safe: bool
    issues: List[CUDAIssue]
    scope_mismatches: List[ScopeMismatch]
    warp_issues: List[WarpSyncIssue]
    memory_issues: List[MemoryOrderingIssue]
    total_threads: int
    total_warps: int
    total_blocks: int

    @property
    def error_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.ERROR)

    @property
    def warning_count(self) -> int:
        return sum(1 for i in self.issues if i.severity == Severity.WARNING)

    def summary(self) -> str:
        status = "✓ SAFE" if self.safe else f"✗ {self.error_count} error(s)"
        return (f"{self.kernel_name}: {status}, "
                f"{self.warning_count} warning(s), "
                f"{self.total_blocks} blocks × {self.block_dim} threads")

    def __repr__(self):
        return f"CUDAValidationResult({self.summary()})"


# ── Parsing Helpers ─────────────────────────────────────────────────

def _extract_kernel_name(code: str) -> str:
    """Extract the kernel function name from CUDA code."""
    m = re.search(r'__global__\s+void\s+(\w+)', code)
    return m.group(1) if m else "unknown_kernel"


def _find_sync_calls(code: str) -> List[Tuple[int, str, SyncScope]]:
    """Find all synchronization calls with line numbers and scopes."""
    results = []
    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        # Check CUDA sync calls
        for func, scope in CUDA_SYNC_SCOPE.items():
            if func in stripped:
                results.append((i, func, scope))
        # Check PTX membar
        for ptx, scope in PTX_MEMBAR_SCOPE.items():
            if ptx in stripped:
                results.append((i, ptx, scope))
    return results


def _find_memory_accesses(code: str) -> List[Tuple[int, str, MemSpace, str]]:
    """Find memory accesses: (line, var, space, access_type: 'load'|'store')."""
    accesses = []
    shared_vars: Set[str] = set()
    # Identify __shared__ declarations
    for i, line in enumerate(code.splitlines(), 1):
        m = re.search(r'__shared__\s+\w+\s+(\w+)', line)
        if m:
            shared_vars.add(m.group(1))

    for i, line in enumerate(code.splitlines(), 1):
        stripped = line.strip()
        if stripped.startswith("//") or stripped.startswith("/*"):
            continue
        # Detect stores (assignments)
        store_match = re.findall(r'(\w+)\s*(?:\[.*?\])?\s*=(?!=)', stripped)
        for var in store_match:
            space = MemSpace.SHARED if var in shared_vars else MemSpace.GLOBAL
            accesses.append((i, var, space, 'store'))
        # Detect loads (rhs references)
        for var in shared_vars:
            if re.search(rf'=\s*.*\b{var}\b', stripped):
                accesses.append((i, var, MemSpace.SHARED, 'load'))

    return accesses


def _find_threadidx_usage(code: str) -> List[Tuple[int, str]]:
    """Find uses of threadIdx, blockIdx, blockDim for divergence analysis."""
    results = []
    for i, line in enumerate(code.splitlines(), 1):
        for idx_var in ['threadIdx', 'blockIdx', 'blockDim', 'gridDim']:
            if idx_var in line:
                results.append((i, idx_var))
    return results


def _find_cooperative_groups(code: str) -> List[Tuple[int, str, SyncScope]]:
    """Find cooperative group usage."""
    results = []
    cg_patterns = {
        r'cooperative_groups::this_thread_block\(\)': SyncScope.BLOCK,
        r'cooperative_groups::this_grid\(\)': SyncScope.DEVICE,
        r'cooperative_groups::tiled_partition<(\d+)>': SyncScope.WARP,
        r'cg::thread_block\b': SyncScope.BLOCK,
        r'cg::grid_group\b': SyncScope.DEVICE,
    }
    for i, line in enumerate(code.splitlines(), 1):
        for pat, scope in cg_patterns.items():
            if re.search(pat, line):
                results.append((i, pat, scope))
    return results


# ── Core Validation Logic ──────────────────────────────────────────

def _check_cross_block_communication(code: str, sync_calls: List) -> List[ScopeMismatch]:
    """Detect cross-block communication protected only by block-scope barriers."""
    mismatches = []

    has_blockidx = bool(re.search(r'blockIdx\.', code))
    has_global_store = bool(re.search(r'(?:g_|d_|global_)\w*\s*\[', code))
    has_global_load = bool(re.search(r'=\s*(?:g_|d_|global_)\w*\s*\[', code))

    block_scoped = [(ln, fn, sc) for ln, fn, sc in sync_calls if sc == SyncScope.BLOCK]
    device_scoped = [(ln, fn, sc) for ln, fn, sc in sync_calls if sc in (SyncScope.DEVICE, SyncScope.SYSTEM)]

    # Pattern: writes to global memory, syncs with block scope, reads from global memory
    if has_blockidx and has_global_store and has_global_load and block_scoped and not device_scoped:
        for ln, fn, sc in block_scoped:
            mismatches.append(ScopeMismatch(
                barrier_type=fn,
                barrier_scope=SyncScope.BLOCK,
                required_scope=SyncScope.DEVICE,
                line=ln,
                pattern="cross_block_mp",
                description=(
                    f"'{fn}' at line {ln} only synchronizes within a thread block, "
                    f"but code accesses global memory with blockIdx-dependent addressing, "
                    f"suggesting cross-block communication."
                ),
                fix=f"Replace '{fn}' with '__threadfence()' + cooperative grid sync, "
                    f"or restructure to avoid cross-block data dependencies.",
            ))

    return mismatches


def _check_cta_vs_device_barriers(code: str, sync_calls: List) -> List[ScopeMismatch]:
    """Check PTX-level membar.cta used where membar.gl is needed."""
    mismatches = []
    has_cross_cta = bool(re.search(r'blockIdx|gridDim|cooperative_groups.*grid', code))
    cta_barriers = [(ln, fn, sc) for ln, fn, sc in sync_calls
                    if fn == "membar.cta" or (fn == "__threadfence_block" and has_cross_cta)]

    if has_cross_cta:
        for ln, fn, sc in cta_barriers:
            mismatches.append(ScopeMismatch(
                barrier_type=fn,
                barrier_scope=SyncScope.BLOCK,
                required_scope=SyncScope.DEVICE,
                line=ln,
                pattern="cta_scope_mismatch",
                description=(
                    f"'{fn}' at line {ln} provides CTA-scope ordering, but the kernel "
                    f"appears to communicate across CTAs via global memory."
                ),
                fix=f"Use 'membar.gl' or '__threadfence()' for device-scope ordering.",
            ))

    return mismatches


def _check_cooperative_group_scope(code: str) -> List[ScopeMismatch]:
    """Check cooperative group scope mismatches."""
    mismatches = []
    cg_uses = _find_cooperative_groups(code)

    has_grid_comm = bool(re.search(r'cooperative_groups.*grid|cg::grid_group', code))
    block_syncs = [(ln, pat, sc) for ln, pat, sc in cg_uses if sc == SyncScope.BLOCK]

    if has_grid_comm and block_syncs:
        for ln, pat, sc in block_syncs:
            mismatches.append(ScopeMismatch(
                barrier_type=f"cooperative_groups block sync",
                barrier_scope=SyncScope.BLOCK,
                required_scope=SyncScope.DEVICE,
                line=ln,
                pattern="cg_scope_mismatch",
                description=(
                    f"Block-scoped cooperative group sync at line {ln} used in a kernel "
                    f"that also uses grid-level cooperative groups — the block sync does "
                    f"not order operations across thread blocks."
                ),
                fix="Use grid_group.sync() for cross-block synchronization.",
            ))

    return mismatches


def _check_warp_divergence(code: str, sync_calls: List) -> List[WarpSyncIssue]:
    """Detect __syncwarp in potentially divergent code paths."""
    issues = []

    lines = code.splitlines()
    in_conditional = False
    conditional_depth = 0
    conditional_start = 0

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Track conditional blocks that depend on threadIdx
        if re.search(r'if\s*\(.*threadIdx', stripped):
            in_conditional = True
            conditional_depth += 1
            conditional_start = i
        elif in_conditional:
            if '{' in stripped:
                conditional_depth += stripped.count('{') - stripped.count('}')
            if '}' in stripped:
                conditional_depth -= 1
                if conditional_depth <= 0:
                    in_conditional = False

        # Check for sync inside conditional
        if in_conditional:
            for func, scope in CUDA_SYNC_SCOPE.items():
                if func in stripped:
                    if func == "__syncwarp":
                        # Check if mask is specified
                        mask_match = re.search(r'__syncwarp\s*\(\s*(\w+)', stripped)
                        if not mask_match or mask_match.group(1) == "":
                            issues.append(WarpSyncIssue(
                                issue_type='divergent_sync',
                                line=i,
                                description=(
                                    f"__syncwarp() at line {i} inside threadIdx-dependent "
                                    f"conditional (from line {conditional_start}) — threads "
                                    f"not taking this branch will not participate in the sync."
                                ),
                                fix="Provide an explicit active mask: __syncwarp(mask) "
                                    "where mask reflects which lanes are active.",
                                warp_mask=None,
                            ))
                    elif func == "__syncthreads":
                        issues.append(WarpSyncIssue(
                            issue_type='divergent_sync',
                            line=i,
                            description=(
                                f"__syncthreads() at line {i} inside threadIdx-dependent "
                                f"conditional — undefined behavior if not all threads "
                                f"in the block reach this barrier."
                            ),
                            fix="Ensure all threads in the block reach __syncthreads(), "
                                "or restructure to avoid conditional barrier.",
                        ))

    return issues


def _check_syncwarp_usage(code: str) -> List[WarpSyncIssue]:
    """Check for correct __syncwarp usage patterns."""
    issues = []
    lines = code.splitlines()

    for i, line in enumerate(lines, 1):
        stripped = line.strip()

        # Deprecated __any, __all, __ballot without _sync suffix
        deprecated = re.findall(r'\b(__any|__all|__ballot)\s*\(', stripped)
        for func in deprecated:
            issues.append(WarpSyncIssue(
                issue_type='deprecated',
                line=i,
                description=f"'{func}()' at line {i} is deprecated since CUDA 9.0.",
                fix=f"Use '{func}_sync(mask, ...)' with an explicit warp mask.",
            ))

        # __shfl without _sync
        shfl_deprecated = re.findall(r'\b(__shfl(?:_up|_down|_xor)?)\s*\(', stripped)
        for func in shfl_deprecated:
            if '_sync' not in func:
                issues.append(WarpSyncIssue(
                    issue_type='deprecated',
                    line=i,
                    description=f"'{func}()' at line {i} is deprecated since CUDA 9.0.",
                    fix=f"Use '{func}_sync(mask, ...)' with an explicit warp mask.",
                ))

        # __syncwarp with hardcoded 0xFFFFFFFF when block size isn't multiple of 32
        if '__syncwarp(0xFFFFFFFF)' in stripped or '__syncwarp(0xffffffff)' in stripped:
            issues.append(WarpSyncIssue(
                issue_type='partial_warp',
                line=i,
                description=(
                    f"__syncwarp(0xFFFFFFFF) at line {i} assumes all 32 lanes are active. "
                    f"This is incorrect if block size is not a multiple of 32 or if "
                    f"called inside divergent code."
                ),
                fix="Compute the active mask dynamically: __activemask() or use "
                    "__ballot_sync(0xFFFFFFFF, 1) to determine active lanes.",
                warp_mask="0xFFFFFFFF",
            ))

    return issues


def _check_shared_memory_ordering(code: str, sync_calls: List) -> List[MemoryOrderingIssue]:
    """Check shared memory access ordering."""
    issues = []
    accesses = _find_memory_accesses(code)

    shared_stores = [(ln, var) for ln, var, sp, at in accesses
                     if sp == MemSpace.SHARED and at == 'store']
    shared_loads = [(ln, var) for ln, var, sp, at in accesses
                    if sp == MemSpace.SHARED and at == 'load']

    sync_lines = {ln for ln, _, _ in sync_calls}

    # Check for shared memory store followed by load without intervening sync
    for s_ln, s_var in shared_stores:
        for l_ln, l_var in shared_loads:
            if l_ln > s_ln and s_var == l_var:
                # Check if there's a sync between them
                has_sync = any(s_ln < sl < l_ln for sl in sync_lines)
                if not has_sync:
                    issues.append(MemoryOrderingIssue(
                        mem_space=MemSpace.SHARED,
                        issue_type='missing_fence',
                        line=l_ln,
                        description=(
                            f"Shared variable '{s_var}' stored at line {s_ln} and loaded "
                            f"at line {l_ln} without an intervening __syncthreads(). "
                            f"Another thread's store may not be visible."
                        ),
                        fix=f"Add __syncthreads() between line {s_ln} and {l_ln}.",
                    ))
                    break  # one issue per store-load pair

    return issues


def _check_global_memory_ordering(code: str, sync_calls: List) -> List[MemoryOrderingIssue]:
    """Check global memory access ordering across blocks."""
    issues = []

    # Pattern: flag-based inter-block communication without threadfence
    flag_stores = list(re.finditer(r'(flag|ready|done|signal)\s*\[.*?\]\s*=', code))
    has_threadfence = '__threadfence' in code and '__threadfence_block' not in code.replace('__threadfence()', '')

    if flag_stores and not has_threadfence:
        for m in flag_stores:
            line = code[:m.start()].count('\n') + 1
            issues.append(MemoryOrderingIssue(
                mem_space=MemSpace.GLOBAL,
                issue_type='missing_fence',
                line=line,
                description=(
                    f"Flag variable written at line {line} without __threadfence(). "
                    f"Data stores before the flag may not be visible to other blocks "
                    f"when they observe the flag."
                ),
                fix="Add __threadfence() between data stores and the flag store.",
            ))

    # Pattern: volatile needed for spin-wait on global memory
    spin_waits = list(re.finditer(r'while\s*\(.*?(flag|ready|done|signal)\s*\[', code))
    for m in spin_waits:
        var = m.group(1)
        line = code[:m.start()].count('\n') + 1
        has_volatile = bool(re.search(rf'volatile\s+.*\b{var}\b', code))
        if not has_volatile:
            issues.append(MemoryOrderingIssue(
                mem_space=MemSpace.GLOBAL,
                issue_type='volatile_needed',
                line=line,
                description=(
                    f"Spin-wait on '{var}' at line {line} without volatile qualifier. "
                    f"The compiler may cache the value in a register, creating an "
                    f"infinite loop."
                ),
                fix=f"Declare '{var}' as volatile, or use cuda::atomic with "
                    f"appropriate memory order.",
            ))

    return issues


# ── Public API ──────────────────────────────────────────────────────

def validate_cuda_kernel(
    code: str,
    grid_dim: Tuple[int, ...] = (1, 1, 1),
    block_dim: Tuple[int, ...] = (32, 1, 1),
) -> CUDAValidationResult:
    """
    Validate a CUDA kernel for thread safety, scope mismatches, warp sync,
    and memory ordering issues.

    Args:
        code: CUDA kernel source code (as string).
        grid_dim: Grid dimensions (blocks).
        block_dim: Block dimensions (threads per block).

    Returns:
        CUDAValidationResult with all detected issues.
    """
    kernel_name = _extract_kernel_name(code)
    total_threads_per_block = block_dim[0] * (block_dim[1] if len(block_dim) > 1 else 1) * (block_dim[2] if len(block_dim) > 2 else 1)
    total_blocks = grid_dim[0] * (grid_dim[1] if len(grid_dim) > 1 else 1) * (grid_dim[2] if len(grid_dim) > 2 else 1)
    total_threads = total_threads_per_block * total_blocks
    total_warps = (total_threads_per_block + WARP_SIZE - 1) // WARP_SIZE * total_blocks

    sync_calls = _find_sync_calls(code)

    # Run all checks
    scope_mismatches = detect_scope_mismatch(code)
    warp_issues = check_warp_sync(code)
    memory_issues = verify_memory_ordering(code)

    # Aggregate into CUDAIssue list
    all_issues: List[CUDAIssue] = []

    for sm in scope_mismatches:
        all_issues.append(CUDAIssue(
            severity=Severity.ERROR if sm.is_critical else Severity.WARNING,
            category='scope_mismatch',
            line=sm.line,
            description=sm.description,
            fix=sm.fix,
            affected_threads="cross-block",
        ))

    for wi in warp_issues:
        sev = Severity.ERROR if wi.issue_type == 'divergent_sync' else Severity.WARNING
        all_issues.append(CUDAIssue(
            severity=sev,
            category='warp_sync',
            line=wi.line,
            description=wi.description,
            fix=wi.fix,
            affected_threads="intra-warp",
        ))

    for mi in memory_issues:
        sev = Severity.ERROR if mi.issue_type in ('missing_fence', 'racy_access') else Severity.WARNING
        all_issues.append(CUDAIssue(
            severity=sev,
            category='memory_ordering',
            line=mi.line,
            description=mi.description,
            fix=mi.fix,
            affected_threads="cross-block" if mi.mem_space == MemSpace.GLOBAL else "intra-block",
        ))

    # Check block size vs warp size
    if total_threads_per_block % WARP_SIZE != 0:
        all_issues.append(CUDAIssue(
            severity=Severity.WARNING,
            category='thread_safety',
            line=None,
            description=(
                f"Block size ({total_threads_per_block}) is not a multiple of warp size "
                f"({WARP_SIZE}). Last warp will have inactive lanes, which affects "
                f"warp-level primitives."
            ),
            fix="Use block sizes that are multiples of 32, or handle partial warps explicitly.",
            affected_threads="intra-warp",
        ))

    safe = not any(i.severity == Severity.ERROR for i in all_issues)

    return CUDAValidationResult(
        kernel_name=kernel_name,
        grid_dim=grid_dim,
        block_dim=block_dim,
        safe=safe,
        issues=all_issues,
        scope_mismatches=scope_mismatches,
        warp_issues=warp_issues,
        memory_issues=memory_issues,
        total_threads=total_threads,
        total_warps=total_warps,
        total_blocks=total_blocks,
    )


def detect_scope_mismatch(code: str) -> List[ScopeMismatch]:
    """
    Detect scope mismatch bugs in CUDA kernel code.

    Checks for:
    - __syncthreads() used for cross-block communication
    - membar.cta where membar.gl is needed
    - Cooperative group block sync for grid-level communication

    Args:
        code: CUDA kernel source code.

    Returns:
        List of ScopeMismatch issues.
    """
    sync_calls = _find_sync_calls(code)
    mismatches = []
    mismatches.extend(_check_cross_block_communication(code, sync_calls))
    mismatches.extend(_check_cta_vs_device_barriers(code, sync_calls))
    mismatches.extend(_check_cooperative_group_scope(code))
    return mismatches


def check_warp_sync(code: str) -> List[WarpSyncIssue]:
    """
    Check warp-level synchronization correctness.

    Detects:
    - __syncwarp() or __syncthreads() inside divergent code
    - Deprecated warp primitives (__any, __all, __ballot, __shfl without _sync)
    - Hardcoded full warp masks that may be incorrect

    Args:
        code: CUDA kernel source code.

    Returns:
        List of WarpSyncIssue issues.
    """
    sync_calls = _find_sync_calls(code)
    issues = []
    issues.extend(_check_warp_divergence(code, sync_calls))
    issues.extend(_check_syncwarp_usage(code))
    return issues


def verify_memory_ordering(
    code: str,
    mem_space: str = "all",
) -> List[MemoryOrderingIssue]:
    """
    Verify memory ordering correctness for shared and/or global memory.

    Checks:
    - Shared memory store-load without __syncthreads()
    - Global memory flag patterns without __threadfence()
    - Spin-waits without volatile

    Args:
        code: CUDA kernel source code.
        mem_space: "shared", "global", or "all" (default).

    Returns:
        List of MemoryOrderingIssue issues.
    """
    sync_calls = _find_sync_calls(code)
    issues = []

    if mem_space in ("shared", "all"):
        issues.extend(_check_shared_memory_ordering(code, sync_calls))
    if mem_space in ("global", "all"):
        issues.extend(_check_global_memory_ordering(code, sync_calls))

    return issues


def analyze_kernel_complexity(code: str) -> Dict:
    """
    Analyze the synchronization complexity of a CUDA kernel.

    Returns metrics about the kernel's concurrency characteristics.
    """
    sync_calls = _find_sync_calls(code)
    accesses = _find_memory_accesses(code)
    cg_uses = _find_cooperative_groups(code)

    shared_accesses = [(ln, var) for ln, var, sp, _ in accesses if sp == MemSpace.SHARED]
    global_accesses = [(ln, var) for ln, var, sp, _ in accesses if sp == MemSpace.GLOBAL]

    scopes_used = set(sc for _, _, sc in sync_calls)
    max_scope = max(scopes_used, key=lambda s: FENCE_LATENCY[s]) if scopes_used else None

    return {
        "kernel_name": _extract_kernel_name(code),
        "sync_call_count": len(sync_calls),
        "sync_scopes": [s.value for s in scopes_used],
        "max_sync_scope": max_scope.value if max_scope else None,
        "shared_memory_accesses": len(shared_accesses),
        "global_memory_accesses": len(global_accesses),
        "cooperative_groups_used": len(cg_uses) > 0,
        "uses_warp_primitives": bool(re.search(r'__syncwarp|__shfl|__ballot|__any|__all', code)),
        "estimated_sync_cost": sum(FENCE_LATENCY[sc] for _, _, sc in sync_calls),
    }


# ── CLI ─────────────────────────────────────────────────────────────

def _main():
    """CLI entry point for CUDA checker."""
    import argparse
    parser = argparse.ArgumentParser(description="LITMUS∞ CUDA Checker")
    parser.add_argument("file", help="CUDA source file (.cu)")
    parser.add_argument("--grid", type=int, nargs=3, default=[1, 1, 1],
                        help="Grid dimensions (default: 1 1 1)")
    parser.add_argument("--block", type=int, nargs=3, default=[32, 1, 1],
                        help="Block dimensions (default: 32 1 1)")
    parser.add_argument("--mem-space", choices=["shared", "global", "all"],
                        default="all", help="Memory space to check")
    parser.add_argument("--json", action="store_true", help="JSON output")
    args = parser.parse_args()

    with open(args.file) as f:
        code = f.read()

    result = validate_cuda_kernel(code, tuple(args.grid), tuple(args.block))

    if args.json:
        import json
        out = {
            "kernel": result.kernel_name,
            "safe": result.safe,
            "errors": result.error_count,
            "warnings": result.warning_count,
            "issues": [
                {"severity": i.severity.value, "category": i.category,
                 "line": i.line, "description": i.description, "fix": i.fix}
                for i in result.issues
            ],
        }
        print(json.dumps(out, indent=2))
    else:
        print(result.summary())
        print()
        for issue in result.issues:
            icon = "✗" if issue.severity == Severity.ERROR else "⚠"
            print(f"  {icon} [{issue.category}] line {issue.line}: {issue.description}")
            print(f"    Fix: {issue.fix}")
            print()

        complexity = analyze_kernel_complexity(code)
        print(f"Complexity: {complexity['sync_call_count']} sync calls, "
              f"scopes: {complexity['sync_scopes']}, "
              f"est. cost: {complexity['estimated_sync_cost']}")


if __name__ == "__main__":
    _main()
