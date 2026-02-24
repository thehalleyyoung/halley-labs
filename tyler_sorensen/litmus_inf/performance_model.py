"""
Performance modelling for fences, barriers, and synchronization primitives.

Estimates costs, optimizes barrier placement, compares synchronization
strategies, and detects false sharing.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re


@dataclass
class CostEstimate:
    fence_type: str
    arch: str
    latency_cycles: int
    throughput_ns: float
    stalls_pipeline: bool
    flushes_store_buffer: bool
    notes: str = ""

    def __str__(self) -> str:
        return (
            f"{self.fence_type} on {self.arch}: "
            f"~{self.latency_cycles} cycles ({self.throughput_ns:.1f} ns), "
            f"pipeline stall={self.stalls_pipeline}, "
            f"store-buffer flush={self.flushes_store_buffer}"
        )


@dataclass
class ComparisonRow:
    strategy: str
    estimated_cycles: int
    throughput_ns: float
    scalability: str
    notes: str = ""


@dataclass
class ComparisonTable:
    arch: str
    rows: List[ComparisonRow] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Synchronization comparison on {self.arch}:"]
        lines.append(f"{'Strategy':<30} {'Cycles':>8} {'ns':>8} {'Scalability':<15} Notes")
        lines.append("-" * 80)
        for r in self.rows:
            lines.append(
                f"{r.strategy:<30} {r.estimated_cycles:>8} {r.throughput_ns:>8.1f} "
                f"{r.scalability:<15} {r.notes}"
            )
        return "\n".join(lines)


@dataclass
class CacheLineReport:
    cache_line_size: int = 64
    variables: List[Dict] = field(default_factory=list)
    false_sharing_pairs: List[Tuple[str, str, str]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        lines = [f"Cache line analysis (line size: {self.cache_line_size}B):"]
        if self.false_sharing_pairs:
            lines.append(f"  False sharing detected ({len(self.false_sharing_pairs)} pair(s)):")
            for v1, v2, reason in self.false_sharing_pairs:
                lines.append(f"    '{v1}' <-> '{v2}': {reason}")
        else:
            lines.append("  No false sharing detected.")
        for rec in self.recommendations:
            lines.append(f"  Recommendation: {rec}")
        return "\n".join(lines)


# ---------- cost database ----------

_FENCE_COSTS: Dict[str, Dict[str, Dict]] = {
    "x86": {
        "mfence": {"cycles": 33, "ns": 10.0, "stall": True, "flush": True,
                    "notes": "Full barrier; flushes store buffer"},
        "sfence": {"cycles": 8, "ns": 2.5, "stall": False, "flush": True,
                   "notes": "Store fence; orders stores only"},
        "lfence": {"cycles": 4, "ns": 1.2, "stall": True, "flush": False,
                   "notes": "Load fence; serialises load pipeline"},
        "lock_prefix": {"cycles": 20, "ns": 6.0, "stall": True, "flush": True,
                        "notes": "LOCK prefix on RMW (implicit mfence on x86)"},
        "xchg": {"cycles": 22, "ns": 7.0, "stall": True, "flush": True,
                 "notes": "Implicit LOCK; full barrier semantics"},
        "cmpxchg": {"cycles": 25, "ns": 8.0, "stall": True, "flush": True,
                    "notes": "Compare-and-swap with LOCK prefix"},
    },
    "arm64": {
        "dmb_ish": {"cycles": 40, "ns": 15.0, "stall": True, "flush": True,
                    "notes": "Full barrier; inner-shareable domain"},
        "dmb_ishst": {"cycles": 25, "ns": 9.0, "stall": False, "flush": True,
                      "notes": "Store barrier; inner-shareable"},
        "dmb_ishld": {"cycles": 18, "ns": 6.5, "stall": True, "flush": False,
                      "notes": "Load barrier; inner-shareable"},
        "dsb_ish": {"cycles": 50, "ns": 18.0, "stall": True, "flush": True,
                    "notes": "Data sync barrier; stronger than DMB"},
        "isb": {"cycles": 60, "ns": 22.0, "stall": True, "flush": True,
                "notes": "Instruction sync barrier; flushes pipeline"},
        "stlr": {"cycles": 5, "ns": 1.8, "stall": False, "flush": True,
                 "notes": "Store-release; lightweight one-way fence"},
        "ldar": {"cycles": 4, "ns": 1.5, "stall": False, "flush": False,
                 "notes": "Load-acquire; lightweight one-way fence"},
    },
    "arm": {
        "dmb_ish": {"cycles": 45, "ns": 18.0, "stall": True, "flush": True,
                    "notes": "Full barrier; ARMv7"},
        "dmb_ishst": {"cycles": 30, "ns": 12.0, "stall": False, "flush": True,
                      "notes": "Store-only barrier"},
        "dsb": {"cycles": 55, "ns": 22.0, "stall": True, "flush": True,
                "notes": "Data sync barrier"},
    },
    "riscv": {
        "fence_rw_rw": {"cycles": 35, "ns": 12.0, "stall": True, "flush": True,
                        "notes": "Full fence; RVWMO"},
        "fence_r_r": {"cycles": 12, "ns": 4.0, "stall": True, "flush": False,
                      "notes": "Read-read fence"},
        "fence_w_w": {"cycles": 10, "ns": 3.5, "stall": False, "flush": True,
                      "notes": "Write-write fence"},
        "fence_tso": {"cycles": 8, "ns": 2.8, "stall": True, "flush": True,
                      "notes": "TSO fence; lighter than full fence"},
        "amoswap": {"cycles": 30, "ns": 10.0, "stall": True, "flush": True,
                    "notes": "Atomic swap; aq/rl variants available"},
    },
    "power": {
        "sync": {"cycles": 50, "ns": 20.0, "stall": True, "flush": True,
                 "notes": "Heavy-weight sync; full barrier"},
        "lwsync": {"cycles": 30, "ns": 12.0, "stall": True, "flush": True,
                   "notes": "Light-weight sync; allows StoreLoad reordering"},
        "isync": {"cycles": 20, "ns": 8.0, "stall": True, "flush": False,
                  "notes": "Instruction sync; context-synchronising"},
        "eieio": {"cycles": 15, "ns": 6.0, "stall": False, "flush": True,
                  "notes": "Enforce in-order execution of I/O; store-store fence"},
    },
}

_FENCE_ALIASES: Dict[str, Tuple[str, str]] = {
    "memory_order_seq_cst": ("x86", "mfence"),
    "memory_order_release": ("x86", "sfence"),
    "memory_order_acquire": ("x86", "lfence"),
    "seq_cst": ("x86", "mfence"),
    "release": ("x86", "sfence"),
    "acquire": ("x86", "lfence"),
    "acq_rel": ("x86", "mfence"),
    "__sync_synchronize": ("x86", "mfence"),
    "atomic_thread_fence": ("x86", "mfence"),
    "__syncthreads": ("arm64", "dmb_ish"),
    "dmb ish": ("arm64", "dmb_ish"),
    "dmb ishst": ("arm64", "dmb_ishst"),
    "dmb ishld": ("arm64", "dmb_ishld"),
    "fence rw,rw": ("riscv", "fence_rw_rw"),
    "fence.tso": ("riscv", "fence_tso"),
}

_SYNC_STRATEGIES: Dict[str, Dict[str, int]] = {
    "mutex": {"base_cycles": 80, "contention_factor": 5, "scalability": 1},
    "spinlock": {"base_cycles": 20, "contention_factor": 20, "scalability": 2},
    "ticket_lock": {"base_cycles": 25, "contention_factor": 8, "scalability": 3},
    "mcs_lock": {"base_cycles": 30, "contention_factor": 3, "scalability": 5},
    "rwlock": {"base_cycles": 50, "contention_factor": 4, "scalability": 4},
    "seqlock": {"base_cycles": 10, "contention_factor": 2, "scalability": 5},
    "rcu": {"base_cycles": 5, "contention_factor": 1, "scalability": 5},
    "cas_loop": {"base_cycles": 25, "contention_factor": 10, "scalability": 3},
    "fetch_add": {"base_cycles": 20, "contention_factor": 6, "scalability": 4},
}

_SCALABILITY_LABELS = {1: "poor", 2: "fair", 3: "moderate", 4: "good", 5: "excellent"}

_TYPE_SIZES = {
    "int": 4, "long": 8, "float": 4, "double": 8,
    "char": 1, "short": 2, "bool": 1, "size_t": 8,
    "uint8_t": 1, "uint16_t": 2, "uint32_t": 4, "uint64_t": 8,
    "int8_t": 1, "int16_t": 2, "int32_t": 4, "int64_t": 8,
    "atomic_int": 4, "atomic_long": 8, "atomic_bool": 1,
    "std::atomic<int>": 4, "std::atomic<long>": 8,
    "AtomicUsize": 8, "AtomicI32": 4, "AtomicI64": 8, "AtomicBool": 1,
    "void*": 8, "uintptr_t": 8,
}


# ---------- public API ----------

def estimate_fence_cost(fence_type: str, arch: str = "x86") -> CostEstimate:
    """Estimate the performance cost of a fence/barrier instruction.

    Args:
        fence_type: Fence instruction name or C11 memory-order name.
        arch: Target architecture (x86, arm64, arm, riscv, power).

    Returns:
        CostEstimate with latency, throughput, and pipeline effects.
    """
    arch = arch.lower().strip()
    ft = fence_type.strip()

    if ft in _FENCE_ALIASES:
        alias_arch, alias_name = _FENCE_ALIASES[ft]
        if arch not in _FENCE_COSTS:
            arch = alias_arch
        ft = alias_name

    ft_key = ft.replace(" ", "_").replace(",", "_").lower()

    arch_db = _FENCE_COSTS.get(arch, _FENCE_COSTS.get("x86", {}))
    entry = arch_db.get(ft_key)

    if entry is None:
        for key, val in arch_db.items():
            if ft_key in key or key in ft_key:
                entry = val
                ft_key = key
                break

    if entry is None:
        return CostEstimate(
            fence_type=fence_type, arch=arch,
            latency_cycles=30, throughput_ns=10.0,
            stalls_pipeline=True, flushes_store_buffer=True,
            notes=f"No specific data for '{fence_type}' on {arch}; using conservative estimate",
        )

    return CostEstimate(
        fence_type=ft_key, arch=arch,
        latency_cycles=entry["cycles"],
        throughput_ns=entry["ns"],
        stalls_pipeline=entry["stall"],
        flushes_store_buffer=entry["flush"],
        notes=entry.get("notes", ""),
    )


def total_sync_overhead(code: str, arch: str = "x86") -> float:
    """Estimate total synchronization overhead in nanoseconds.

    Scans source for fences, barriers, and atomic RMW operations,
    sums their estimated costs.

    Args:
        code: Source code.
        arch: Target architecture.

    Returns:
        Estimated total synchronization cost in nanoseconds.
    """
    total = 0.0

    fence_patterns = [
        (r"atomic_thread_fence\(\s*memory_order_(\w+)\s*\)", None),
        (r"__sync_synchronize\(\)", "__sync_synchronize"),
        (r'asm\s+volatile\s*\(\s*"mfence"', "mfence"),
        (r'asm\s+volatile\s*\(\s*"sfence"', "sfence"),
        (r'asm\s+volatile\s*\(\s*"lfence"', "lfence"),
        (r'asm\s+volatile\s*\(\s*"dmb ish"', "dmb ish"),
        (r'asm\s+volatile\s*\(\s*"dmb ishst"', "dmb ishst"),
        (r'asm\s+volatile\s*\(\s*"fence rw,rw"', "fence rw,rw"),
        (r'asm\s+volatile\s*\(\s*"sync"', "sync"),
        (r"__syncthreads\(\)", "__syncthreads"),
        (r"fence\(\s*Ordering::(\w+)\s*\)", None),
    ]

    for pattern, fixed_name in fence_patterns:
        for m in re.finditer(pattern, code):
            if fixed_name:
                name = fixed_name
            else:
                name = m.group(1) if m.lastindex else "seq_cst"
            cost = estimate_fence_cost(name, arch)
            total += cost.throughput_ns

    atomic_rmw_patterns = [
        r"atomic_fetch_add",
        r"atomic_fetch_sub",
        r"atomic_compare_exchange",
        r"__sync_val_compare_and_swap",
        r"__sync_fetch_and_add",
        r"InterlockedIncrement",
        r"InterlockedCompareExchange",
        r"\.fetch_add\(",
        r"\.compare_exchange",
    ]
    for pat in atomic_rmw_patterns:
        count = len(re.findall(pat, code))
        if count:
            rmw_cost = estimate_fence_cost("cmpxchg" if "compare" in pat else "lock_prefix", arch)
            total += count * rmw_cost.throughput_ns

    store_release = len(re.findall(r"memory_order_release|Ordering::Release", code))
    load_acquire = len(re.findall(r"memory_order_acquire|Ordering::Acquire", code))
    if arch in ("arm64",):
        total += store_release * 1.8  # stlr
        total += load_acquire * 1.5   # ldar
    elif arch in ("x86",):
        pass  # x86 acquires/releases are free (TSO)

    return total


def optimize_barrier_placement(code: str, arch: str = "x86") -> str:
    """Optimize barrier/fence placement to minimise performance cost.

    Replaces overly-strong barriers with weaker alternatives where safe,
    removes redundant barriers, and applies architecture-specific optimisations.

    Args:
        code: Source code.
        arch: Target architecture.

    Returns:
        Optimised source code.
    """
    result = code
    arch = arch.lower()

    if arch == "x86":
        result = re.sub(
            r"atomic_thread_fence\(\s*memory_order_acquire\s*\)",
            "/* x86: acquire fence elided (TSO provides acquire semantics) */",
            result,
        )
        result = re.sub(
            r"atomic_thread_fence\(\s*memory_order_release\s*\)",
            "/* x86: release fence elided (TSO provides release semantics) */",
            result,
        )

    lines = result.splitlines()
    optimized_lines = []
    prev_fence = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        is_fence = bool(re.search(
            r"atomic_thread_fence|__sync_synchronize|mfence|dmb|fence\s+rw|sync\b",
            stripped,
        ))
        if is_fence and prev_fence is not None and i - prev_fence <= 2:
            only_whitespace = all(
                lines[j].strip() == "" for j in range(prev_fence + 1, i)
            )
            if only_whitespace:
                optimized_lines.append(
                    f"    /* redundant fence removed (merged with line {prev_fence + 1}) */"
                )
                continue
        if is_fence:
            prev_fence = i
        optimized_lines.append(line)

    result = "\n".join(optimized_lines)

    if arch in ("arm64", "arm"):
        result = re.sub(
            r"(atomic_store_explicit\([^,]+,\s*[^,]+,\s*)memory_order_seq_cst(\s*\))",
            r"\1memory_order_release\2  /* arm64: relaxed to release (sufficient for store) */",
            result,
        )
        result = re.sub(
            r"(atomic_load_explicit\([^,]+,\s*)memory_order_seq_cst(\s*\))",
            r"\1memory_order_acquire\2  /* arm64: relaxed to acquire (sufficient for load) */",
            result,
        )

    return result


def compare_sync_strategies(
    strategies: Optional[List[str]] = None,
    arch: str = "x86",
) -> ComparisonTable:
    """Compare synchronization strategies for a given architecture.

    Args:
        strategies: List of strategy names. Default: all known strategies.
        arch: Target architecture.

    Returns:
        ComparisonTable with cost/scalability data.
    """
    if strategies is None:
        strategies = list(_SYNC_STRATEGIES.keys())

    arch = arch.lower()
    arch_factor = {"x86": 1.0, "arm64": 1.3, "arm": 1.5, "riscv": 1.2, "power": 1.4}
    factor = arch_factor.get(arch, 1.0)

    rows: List[ComparisonRow] = []
    for name in strategies:
        info = _SYNC_STRATEGIES.get(name.lower())
        if info is None:
            rows.append(ComparisonRow(
                strategy=name, estimated_cycles=0, throughput_ns=0.0,
                scalability="unknown", notes=f"Unknown strategy '{name}'",
            ))
            continue

        cycles = int(info["base_cycles"] * factor)
        ns = cycles / 3.0  # ~3 GHz
        scal = _SCALABILITY_LABELS.get(info["scalability"], "unknown")

        notes_parts = []
        if info["contention_factor"] > 8:
            notes_parts.append("high contention penalty")
        if info["scalability"] >= 4:
            notes_parts.append("scales well with cores")
        if name == "rcu":
            notes_parts.append("read-side nearly free; write-side expensive")
        if name == "seqlock":
            notes_parts.append("best for read-heavy with rare writes")

        rows.append(ComparisonRow(
            strategy=name,
            estimated_cycles=cycles,
            throughput_ns=round(ns, 1),
            scalability=scal,
            notes="; ".join(notes_parts),
        ))

    rows.sort(key=lambda r: r.estimated_cycles)
    return ComparisonTable(arch=arch, rows=rows)


def cache_line_analysis(code: str) -> CacheLineReport:
    """Analyse code for false sharing issues.

    Identifies variables likely to share a cache line when accessed
    by different threads, which causes performance degradation.

    Args:
        code: Source code (C/C++/Rust).

    Returns:
        CacheLineReport with false sharing pairs and recommendations.
    """
    CACHE_LINE = 64
    report = CacheLineReport(cache_line_size=CACHE_LINE)

    struct_pattern = re.compile(
        r"(?:typedef\s+)?struct\s+(\w+)?\s*\{([^}]+)\}",
        re.DOTALL,
    )

    structs = struct_pattern.findall(code)
    for struct_name, body in structs:
        fields: List[Dict] = []
        offset = 0
        for line in body.splitlines():
            stripped = line.strip().rstrip(";").strip()
            if not stripped or stripped.startswith("//") or stripped.startswith("/*"):
                continue

            aligned = re.search(r"__attribute__\s*\(\s*\(\s*aligned\s*\(\s*(\d+)\s*\)", stripped)
            cacheline_aligned = bool(re.search(
                r"alignas\s*\(\s*64\s*\)|__cacheline_aligned|CACHE_ALIGNED", stripped
            ))

            for type_name, size in _TYPE_SIZES.items():
                if type_name in stripped:
                    m = re.search(rf"(?:{re.escape(type_name)})\s+(\w+)", stripped)
                    if m:
                        var_name = m.group(1)
                        if aligned:
                            align_val = int(aligned.group(1))
                            offset = ((offset + align_val - 1) // align_val) * align_val
                        if cacheline_aligned:
                            offset = ((offset + CACHE_LINE - 1) // CACHE_LINE) * CACHE_LINE

                        field_info = {
                            "name": var_name,
                            "type": type_name,
                            "size": size,
                            "offset": offset,
                            "cache_line": offset // CACHE_LINE,
                            "struct": struct_name or "<anonymous>",
                        }
                        fields.append(field_info)
                        report.variables.append(field_info)
                        offset += size
                    break

        thread_affinity: Dict[str, set] = {}
        for f in fields:
            fname = f["name"]
            patterns = [
                (rf"thread\d*.*\b{re.escape(fname)}\b", "different_threads"),
                (rf"\b{re.escape(fname)}\b.*atomic", "atomic"),
                (rf"atomic.*\b{re.escape(fname)}\b", "atomic"),
            ]
            for pat, label in patterns:
                if re.search(pat, code, re.IGNORECASE):
                    thread_affinity.setdefault(fname, set()).add(label)

        for i in range(len(fields)):
            for j in range(i + 1, len(fields)):
                f1, f2 = fields[i], fields[j]
                if f1["cache_line"] == f2["cache_line"]:
                    either_threaded = (
                        f1["name"] in thread_affinity or f2["name"] in thread_affinity
                    )
                    either_atomic = (
                        "atomic" in f1["type"].lower() or "atomic" in f2["type"].lower()
                    )

                    if either_threaded or either_atomic:
                        reason = (
                            f"Both at cache line {f1['cache_line']} "
                            f"(offsets {f1['offset']} and {f2['offset']}) "
                            f"in struct {f1['struct']}"
                        )
                        report.false_sharing_pairs.append(
                            (f1["name"], f2["name"], reason)
                        )

    global_vars: List[Dict] = []
    global_pat = re.compile(
        r"^(?:static\s+|extern\s+|volatile\s+|_Atomic\s+)*"
        r"(atomic_\w+|std::atomic<\w+>|\w+)\s+(\w+)\s*(?:=|;)",
        re.MULTILINE,
    )
    for m in global_pat.finditer(code):
        type_name = m.group(1)
        var_name = m.group(2)
        size = _TYPE_SIZES.get(type_name, 4)
        global_vars.append({"name": var_name, "type": type_name, "size": size})

    if len(global_vars) >= 2:
        offset = 0
        for gv in global_vars:
            gv["offset"] = offset
            gv["cache_line"] = offset // CACHE_LINE
            offset += gv["size"]

        for i in range(len(global_vars)):
            for j in range(i + 1, len(global_vars)):
                g1, g2 = global_vars[i], global_vars[j]
                if g1["cache_line"] == g2["cache_line"]:
                    if "atomic" in g1["type"].lower() or "atomic" in g2["type"].lower():
                        reason = (
                            f"Adjacent globals may share cache line "
                            f"(estimated offsets {g1['offset']} and {g2['offset']})"
                        )
                        report.false_sharing_pairs.append(
                            (g1["name"], g2["name"], reason)
                        )

    if report.false_sharing_pairs:
        report.recommendations.append(
            "Pad or align hot fields to cache-line boundaries: "
            "__attribute__((aligned(64))) or alignas(64)"
        )
        report.recommendations.append(
            "Consider grouping thread-local fields together and padding between groups"
        )
        report.recommendations.append(
            "Use __cacheline_aligned_in_smp (Linux) or equivalent for per-CPU data"
        )

    return report
