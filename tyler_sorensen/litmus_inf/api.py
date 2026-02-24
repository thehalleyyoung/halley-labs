#!/usr/bin/env python3
"""
LITMUS∞ Public API — programmatic access to cross-platform memory model checking.

Usage:
    from api import check_portability, find_fence_bugs, minimize_fences
    result = check_portability("mp", "x86", "arm")
    bugs = find_fence_bugs("mp", architecture="arm")
"""

import sys
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(__file__))
from portcheck import (
    MemOp, LitmusTest, PATTERNS, ARCHITECTURES,
    compute_joint_automorphisms, compute_orbits,
    verify_test, recommend_fence, _identify_per_thread_violations,
    _is_scope_mismatch_pattern, _is_gpu_model,
    check_portability as _check_portability_raw,
)


# ── Result Dataclasses ──────────────────────────────────────────────

@dataclass
class PortabilityResult:
    """Result of checking whether code is portable between two architectures."""
    test_name: str
    source_arch: str
    target_arch: str
    safe: bool
    forbidden_outcome: Dict[str, int]
    fence_fix: Optional[str]
    compression_ratio: float
    orbits_checked: int
    total_outcomes: int
    scope_mismatch: bool = False
    explanation: str = ""

    def __repr__(self):
        status = "✓ SAFE" if self.safe else "✗ FAIL"
        s = f"PortabilityResult({self.test_name}: {self.source_arch}→{self.target_arch} = {status}"
        if self.fence_fix:
            s += f", fix: {self.fence_fix}"
        return s + ")"


@dataclass
class FenceBug:
    """A detected fence insufficiency or scope mismatch."""
    pattern: str
    architecture: str
    thread: int
    bug_type: str  # 'missing_fence', 'wrong_scope', 'insufficient_ordering'
    severity: str  # 'critical', 'warning'
    description: str
    fix: str
    affected_ops: List[str] = field(default_factory=list)

    def __repr__(self):
        return f"FenceBug({self.pattern}@{self.architecture} T{self.thread}: {self.bug_type} [{self.severity}])"


@dataclass
class OptimizedCode:
    """Result of fence minimization — per-thread minimal fences."""
    test_name: str
    target_arch: str
    original_fence_cost: float
    optimized_fence_cost: float
    savings_pct: float
    per_thread_fences: Dict[int, str]
    safe: bool
    explanation: str = ""

    def __repr__(self):
        return f"OptimizedCode({self.test_name}@{self.target_arch}: {self.savings_pct:.0f}% savings, safe={self.safe})"


@dataclass
class ArchComparisonRow:
    """A single row in the architecture comparison table."""
    pattern: str
    results: Dict[str, bool]  # arch -> safe
    fence_fixes: Dict[str, Optional[str]]  # arch -> fix

@dataclass
class ArchComparisonTable:
    """Full comparison table across architectures."""
    patterns: List[str]
    architectures: List[str]
    rows: List[ArchComparisonRow]
    scope_mismatches: List[str]
    total_safe: int
    total_fail: int

    def __repr__(self):
        return (f"ArchComparisonTable({len(self.patterns)} tests × {len(self.architectures)} archs, "
                f"safe={self.total_safe}, fail={self.total_fail}, scope_mismatches={len(self.scope_mismatches)})")

    def to_text(self) -> str:
        """Render as a formatted text table."""
        hdr = f"{'Pattern':<40}" + "".join(f"{a:>8}" for a in self.architectures)
        lines = [hdr, "-" * len(hdr)]
        for row in self.rows:
            cells = []
            for arch in self.architectures:
                cells.append("  ✓Safe" if row.results.get(arch, False) else "  ✗FAIL")
            lines.append(f"{row.pattern:<40}" + "".join(cells))
        return "\n".join(lines)


@dataclass
class ValidationResult:
    """Result of GPU kernel validation."""
    kernel_name: str
    scope: str
    safe: bool
    scope_mismatches: List[Dict]
    fence_recommendations: List[Dict]
    patterns_checked: int
    explanation: str = ""

    def __repr__(self):
        status = "✓ SAFE" if self.safe else f"✗ {len(self.scope_mismatches)} scope mismatch(es)"
        return f"ValidationResult({self.kernel_name}@{self.scope}: {status})"


# ── Helper ──────────────────────────────────────────────────────────

ALL_ARCHS = list(ARCHITECTURES.keys())

_FENCE_COST = {
    'arm': {'dmb ishst': 1, 'dmb ishld': 2, 'dmb ish': 4},
    'riscv': {'fence r,r': 1, 'fence w,w': 1, 'fence r,w': 1, 'fence w,r': 2, 'fence rw,rw': 4},
}


def _build_test(name: str) -> LitmusTest:
    """Build a LitmusTest from a built-in pattern name."""
    if name not in PATTERNS:
        raise ValueError(f"Unknown pattern '{name}'. Available: {sorted(PATTERNS.keys())}")
    pat = PATTERNS[name]
    n_threads = max(op.thread for op in pat['ops']) + 1
    return LitmusTest(
        name=name, n_threads=n_threads,
        addresses=pat['addresses'], ops=pat['ops'], forbidden=pat['forbidden'],
    )


# ── Public API ──────────────────────────────────────────────────────

def check_portability(test: str, source_arch: str, target_arch: str) -> PortabilityResult:
    """
    Check if a concurrent pattern is safe to port from source_arch to target_arch.

    Args:
        test: Name of a built-in litmus test pattern (e.g., "mp", "sb")
              OR a code string containing concurrent memory operations.
        source_arch: Source architecture (e.g., "x86", "arm", "riscv").
        target_arch: Target architecture (e.g., "arm", "opencl_wg", "ptx_cta").

    Returns:
        PortabilityResult with safety status, fence fix, and compression info.

    Raises:
        ValueError: If architecture names are invalid.
    """
    if source_arch not in ARCHITECTURES:
        raise ValueError(f"Unknown source architecture '{source_arch}'. Available: {ALL_ARCHS}")
    if target_arch not in ARCHITECTURES:
        raise ValueError(f"Unknown target architecture '{target_arch}'. Available: {ALL_ARCHS}")

    # If test is a known pattern name, use it directly; otherwise parse as code
    if test in PATTERNS:
        lt = _build_test(test)
        test_name = test
    else:
        from code_analyzer import get_analyzer
        analyzer = get_analyzer()
        analysis = analyzer.analyze_code(test)
        if not analysis.patterns_found:
            return PortabilityResult(
                test_name="unknown", source_arch=source_arch,
                target_arch=target_arch, safe=True,
                forbidden_outcome={}, fence_fix=None,
                compression_ratio=1.0, orbits_checked=0,
                total_outcomes=0,
                explanation="No concurrent memory patterns detected in code.",
            )
        best = analysis.patterns_found[0]
        lt = _build_test(best.pattern_name)
        test_name = best.pattern_name
    model = ARCHITECTURES[target_arch]

    autos = compute_joint_automorphisms(lt)
    total, n_orbits = compute_orbits(lt, autos)

    if not lt.forbidden:
        safe = True
        n_checked = 0
    else:
        forbidden_allowed, n_checked = verify_test(lt, model)
        safe = not forbidden_allowed

    fence_fix = None
    if not safe:
        fence_fix = recommend_fence(lt, target_arch, model)

    is_scope = _is_scope_mismatch_pattern(lt)
    explanation = ""
    if is_scope and not safe:
        explanation = (
            f"Scope mismatch: workgroup-scope barriers do not provide "
            f"cross-workgroup ordering on {target_arch}."
        )
    elif not safe:
        explanation = f"Pattern '{test_name}' allows the forbidden outcome on {target_arch}."

    return PortabilityResult(
        test_name=test_name,
        source_arch=source_arch,
        target_arch=target_arch,
        safe=safe,
        forbidden_outcome=lt.forbidden,
        fence_fix=fence_fix,
        compression_ratio=total / n_orbits if n_orbits > 0 else 1.0,
        orbits_checked=n_orbits,
        total_outcomes=total,
        scope_mismatch=is_scope and not safe,
        explanation=explanation,
    )


def find_fence_bugs(code: str, architecture: str = "all") -> List[FenceBug]:
    """
    Find fence insufficiency bugs in a pattern or code string.

    Args:
        code: Name of a built-in litmus test pattern, OR a code string
              containing concurrent memory operations (C/C++/CUDA/pseudocode).
        architecture: Target architecture, or "all" for all 10 architectures.

    Returns:
        List of FenceBug objects describing each detected issue.
    """
    # If code is a known pattern name, use it directly; otherwise parse as code
    if code in PATTERNS:
        lt = _build_test(code)
    else:
        from code_analyzer import get_analyzer
        analyzer = get_analyzer()
        analysis = analyzer.analyze_code(code)
        if not analysis.patterns_found:
            return []
        # Use the best matching pattern
        lt = _build_test(analysis.patterns_found[0].pattern_name)
    archs = ALL_ARCHS if architecture == "all" else [architecture]
    bugs: List[FenceBug] = []

    for arch in archs:
        if arch not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{arch}'.")
        model = ARCHITECTURES[arch]

        if not lt.forbidden:
            continue
        forbidden_allowed, _ = verify_test(lt, model)
        if not forbidden_allowed:
            continue

        # Identify per-thread violations
        violations = _identify_per_thread_violations(lt, model)
        fence_rec = recommend_fence(lt, arch, model)
        is_scope = _is_scope_mismatch_pattern(lt)

        for tid, pairs in violations.items():
            if is_scope and _is_gpu_model(model):
                bug_type = 'wrong_scope'
                severity = 'critical'
                desc = (
                    f"Thread {tid} uses workgroup-scope barrier for cross-workgroup "
                    f"communication — provides no ordering guarantee."
                )
                fix = f"Use device-scope barrier on thread {tid}."
            else:
                pair_types = set()
                for before_type, after_type in pairs:
                    pair_types.add(f"{before_type}→{after_type}")

                bug_type = 'missing_fence'
                severity = 'critical' if 'store→load' in pair_types else 'warning'
                desc = f"Thread {tid} has unordered {', '.join(sorted(pair_types))} pairs."
                fix_parts = fence_rec.split(';') if fence_rec else []
                thread_fix = ""
                for part in fix_parts:
                    if f"T{tid}" in part or f"(T{tid})" in part:
                        thread_fix = part.strip()
                        break
                fix = thread_fix or (fence_rec or "Add full barrier.")

            affected = [f"{bt}→{at}" for bt, at in pairs]
            bugs.append(FenceBug(
                pattern=code,
                architecture=arch,
                thread=tid,
                bug_type=bug_type,
                severity=severity,
                description=desc,
                fix=fix,
                affected_ops=affected,
            ))

    return bugs


def minimize_fences(code: str, target_arch: str) -> OptimizedCode:
    """
    Compute per-thread minimal fences for a pattern on a target architecture.

    Compares the cost of minimal per-thread fences vs. a coarse full barrier,
    returning the savings percentage and per-thread fence assignments.

    Args:
        code: Name of a built-in litmus test pattern.
        target_arch: Target architecture (must be "arm" or "riscv" for typed fences).

    Returns:
        OptimizedCode with per-thread fences and savings analysis.
    """
    lt = _build_test(code)
    if target_arch not in ARCHITECTURES:
        raise ValueError(f"Unknown architecture '{target_arch}'.")

    model = ARCHITECTURES[target_arch]
    n_threads = lt.n_threads

    if not lt.forbidden:
        return OptimizedCode(
            test_name=code, target_arch=target_arch,
            original_fence_cost=0, optimized_fence_cost=0,
            savings_pct=0.0, per_thread_fences={}, safe=True,
            explanation="No forbidden outcome to prevent.",
        )

    forbidden_allowed, _ = verify_test(lt, model)
    if not forbidden_allowed:
        return OptimizedCode(
            test_name=code, target_arch=target_arch,
            original_fence_cost=0, optimized_fence_cost=0,
            savings_pct=0.0, per_thread_fences={}, safe=True,
            explanation="Pattern is already safe — no fences needed.",
        )

    fence_rec = recommend_fence(lt, target_arch, model)
    per_thread: Dict[int, str] = {}
    if fence_rec:
        for part in fence_rec.split(';'):
            part = part.strip()
            for t in range(n_threads):
                if f"(T{t})" in part or f"T{t}" in part:
                    fence_name = part.split('(')[0].strip()
                    per_thread[t] = fence_name
                    break

    costs = _FENCE_COST.get(target_arch, {})
    coarse_fence = 'dmb ish' if target_arch == 'arm' else 'fence rw,rw'
    coarse_cost_per = costs.get(coarse_fence, 4)
    original_cost = coarse_cost_per * n_threads

    optimized_cost = sum(costs.get(f, coarse_cost_per) for f in per_thread.values())
    savings = ((original_cost - optimized_cost) / original_cost * 100) if original_cost > 0 else 0.0

    return OptimizedCode(
        test_name=code,
        target_arch=target_arch,
        original_fence_cost=original_cost,
        optimized_fence_cost=optimized_cost,
        savings_pct=savings,
        per_thread_fences=per_thread,
        safe=True,
        explanation=f"Reduced from {n_threads}× {coarse_fence} to per-thread minimal fences.",
    )


def compare_architectures(test: str, archs: List[str] = None) -> ArchComparisonTable:
    """
    Compare a litmus test (or all tests) across multiple architectures.

    Args:
        test: Pattern name, or "all" to compare all built-in patterns.
        archs: List of architecture names (default: all 10 architectures).

    Returns:
        ArchComparisonTable with per-pattern, per-architecture results.
    """
    arch_list = archs or ALL_ARCHS
    for a in arch_list:
        if a not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture '{a}'.")

    patterns = sorted(PATTERNS.keys()) if test == "all" else [test]
    for p in patterns:
        if p not in PATTERNS:
            raise ValueError(f"Unknown pattern '{p}'.")

    rows = []
    scope_mismatches = []
    total_safe = 0
    total_fail = 0

    for pat_name in patterns:
        lt = _build_test(pat_name)
        results_map = {}
        fixes_map = {}

        for arch in arch_list:
            model = ARCHITECTURES[arch]
            if not lt.forbidden:
                results_map[arch] = True
                fixes_map[arch] = None
                total_safe += 1
                continue
            forbidden_allowed, _ = verify_test(lt, model)
            safe = not forbidden_allowed
            results_map[arch] = safe
            fixes_map[arch] = None
            if safe:
                total_safe += 1
            else:
                total_fail += 1
                fixes_map[arch] = recommend_fence(lt, arch, model)

        rows.append(ArchComparisonRow(pattern=pat_name, results=results_map, fence_fixes=fixes_map))

        # Detect scope mismatch
        cpu_archs = [a for a in arch_list if not _is_gpu_model(ARCHITECTURES[a])]
        gpu_archs = [a for a in arch_list if _is_gpu_model(ARCHITECTURES[a])]
        if cpu_archs and gpu_archs:
            all_cpu_safe = all(results_map.get(a, True) for a in cpu_archs)
            any_gpu_fail = any(not results_map.get(a, True) for a in gpu_archs)
            if all_cpu_safe and any_gpu_fail:
                scope_mismatches.append(pat_name)

    return ArchComparisonTable(
        patterns=patterns,
        architectures=arch_list,
        rows=rows,
        scope_mismatches=scope_mismatches,
        total_safe=total_safe,
        total_fail=total_fail,
    )


def validate_gpu_kernel(kernel: str, scope: str = "device") -> ValidationResult:
    """
    Validate a GPU kernel pattern against scope mismatch and ordering bugs.

    Tests the named pattern against all GPU memory models to detect scope mismatches
    and provides fix recommendations.

    Args:
        kernel: Name of a GPU litmus test pattern (e.g., "gpu_mp_wg", "gpu_barrier_scope_mismatch").
        scope: Scope level to validate at ("workgroup" or "device").

    Returns:
        ValidationResult with scope mismatch details and fence recommendations.
    """
    lt = _build_test(kernel)

    gpu_archs = {
        "workgroup": ["opencl_wg", "vulkan_wg", "ptx_cta"],
        "device": ["opencl_dev", "vulkan_dev", "ptx_gpu"],
    }
    target_archs = gpu_archs.get(scope, gpu_archs["device"])

    mismatches = []
    recommendations = []
    all_safe = True

    for arch in target_archs:
        model = ARCHITECTURES[arch]
        if not lt.forbidden:
            continue
        forbidden_allowed, _ = verify_test(lt, model)
        safe = not forbidden_allowed

        if not safe:
            all_safe = False
            fence_rec = recommend_fence(lt, arch, model)

            if _is_scope_mismatch_pattern(lt):
                mismatches.append({
                    "architecture": arch,
                    "type": "scope_mismatch",
                    "description": f"Workgroup-scope barrier insufficient for cross-workgroup communication on {arch}.",
                    "fix": f"Use device-scope barrier ({_device_scope_fix(arch)}).",
                })
            else:
                recommendations.append({
                    "architecture": arch,
                    "fence": fence_rec,
                    "description": f"Missing ordering on {arch}.",
                })

    return ValidationResult(
        kernel_name=kernel,
        scope=scope,
        safe=all_safe,
        scope_mismatches=mismatches,
        fence_recommendations=recommendations,
        patterns_checked=len(target_archs),
        explanation=(
            "All checked GPU models preserve the required ordering."
            if all_safe else
            f"Found {len(mismatches)} scope mismatch(es) and {len(recommendations)} ordering issue(s)."
        ),
    )


def _device_scope_fix(arch: str) -> str:
    """Return the device-scope barrier name for a given GPU architecture."""
    fixes = {
        "opencl_wg": "use work_group_barrier with CLK_GLOBAL_MEM_FENCE at device scope",
        "vulkan_wg": "use device-scope barrier via VkMemoryBarrier",
        "ptx_cta": "use membar.gl instead of membar.cta",
        "opencl_dev": "barrier already at device scope",
        "vulkan_dev": "barrier already at device scope",
        "ptx_gpu": "barrier already at device scope",
    }
    return fixes.get(arch, "use device-scope barrier")


# ── Code Analysis (public API) ──────────────────────────────────────

def analyze_code(code: str, language: str = "auto"):
    """
    Analyze concurrent code to identify litmus test patterns.

    Args:
        code: Code string (C/C++/CUDA/pseudocode) with concurrent memory ops.
        language: Language hint ("c", "cpp", "cuda", "opencl", "pseudo", "auto").

    Returns:
        CodeAnalysisResult with matched patterns and extracted operations.
    """
    from code_analyzer import get_analyzer
    return get_analyzer().analyze_code(code, language)


def check_code(code: str, target_arch: str = None, language: str = "auto"):
    """
    Full pipeline: parse code → match patterns → check portability.

    Args:
        code: Code string with concurrent memory operations.
        target_arch: Target architecture (or None for all).
        language: Language hint.

    Returns:
        List of dicts with pattern, confidence, safety, and fix info.
    """
    from code_analyzer import get_analyzer
    return get_analyzer().check_code(code, target_arch, language)


# ── CLI Entry Point ─────────────────────────────────────────────────

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="LITMUS∞ API — programmatic portability checking")
    sub = parser.add_subparsers(dest="cmd")

    p_port = sub.add_parser("check", help="Check portability between architectures")
    p_port.add_argument("test", help="Pattern name or code string")
    p_port.add_argument("--source", default="x86", help="Source architecture")
    p_port.add_argument("--target", required=True, help="Target architecture")

    p_bugs = sub.add_parser("bugs", help="Find fence bugs")
    p_bugs.add_argument("test", help="Pattern name or code string")
    p_bugs.add_argument("--arch", default="all", help="Architecture (default: all)")

    p_min = sub.add_parser("minimize", help="Minimize fences")
    p_min.add_argument("test", help="Pattern name")
    p_min.add_argument("--target", required=True, help="Target architecture")

    p_cmp = sub.add_parser("compare", help="Compare architectures")
    p_cmp.add_argument("test", help="Pattern name or 'all'")
    p_cmp.add_argument("--archs", nargs="*", help="Architectures to compare")

    p_gpu = sub.add_parser("validate-gpu", help="Validate GPU kernel")
    p_gpu.add_argument("kernel", help="GPU pattern name")
    p_gpu.add_argument("--scope", default="device", choices=["workgroup", "device"])

    p_analyze = sub.add_parser("analyze", help="Analyze code for patterns")
    p_analyze.add_argument("code", help="Code string or @filename")

    args = parser.parse_args()

    if args.cmd == "check":
        result = check_portability(args.test, args.source, args.target)
        print(result)
        if result.fence_fix:
            print(f"  Fix: {result.fence_fix}")
    elif args.cmd == "bugs":
        bugs = find_fence_bugs(args.test, args.arch)
        for b in bugs:
            print(b)
            print(f"  {b.description}")
            print(f"  Fix: {b.fix}")
    elif args.cmd == "minimize":
        opt = minimize_fences(args.test, args.target)
        print(opt)
        for tid, fence in sorted(opt.per_thread_fences.items()):
            print(f"  T{tid}: {fence}")
    elif args.cmd == "compare":
        table = compare_architectures(args.test, args.archs)
        print(table.to_text())
        if table.scope_mismatches:
            print(f"\nScope mismatches: {', '.join(table.scope_mismatches)}")
    elif args.cmd == "validate-gpu":
        result = validate_gpu_kernel(args.kernel, args.scope)
        print(result)
        for m in result.scope_mismatches:
            print(f"  ✗ {m['architecture']}: {m['description']}")
        for r in result.fence_recommendations:
            print(f"  → {r['architecture']}: {r['fence']}")
    else:
        parser.print_help()


if __name__ == "__main__":
    _main()
