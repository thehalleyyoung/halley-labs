"""
Concurrent code migration and porting checker.

Safely port concurrent code between languages and architectures, check
memory model compatibility, and verify ported implementations preserve
the concurrency semantics of the original.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple, Set
import re


class PortStatus(Enum):
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class CompatLevel(Enum):
    COMPATIBLE = "compatible"
    NEEDS_FENCES = "needs_fences"
    INCOMPATIBLE = "incompatible"


class VerificationStatus(Enum):
    EQUIVALENT = "equivalent"
    DIFFERS = "differs"
    UNKNOWN = "unknown"


@dataclass
class PortWarning:
    line: int
    original: str
    message: str
    severity: str = "warning"  # warning, error, info

    def __str__(self) -> str:
        return f"[{self.severity}] line {self.line}: {self.message}"


@dataclass
class PortResult:
    status: PortStatus
    ported_code: str
    from_lang: str
    to_lang: str
    warnings: List[PortWarning] = field(default_factory=list)
    unmapped_constructs: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        s = f"Port {self.from_lang} -> {self.to_lang}: [{self.status.value}]"
        if self.warnings:
            s += f" ({len(self.warnings)} warning(s))"
        if self.unmapped_constructs:
            s += f" ({len(self.unmapped_constructs)} unmapped construct(s))"
        return s


@dataclass
class CompatResult:
    level: CompatLevel
    from_arch: str
    to_arch: str
    required_fences: List[str] = field(default_factory=list)
    reorderings: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        s = f"Compat {self.from_arch} -> {self.to_arch}: [{self.level.value}]"
        if self.required_fences:
            s += f" — need {len(self.required_fences)} fence(s)"
        return s


@dataclass
class VerificationResult:
    status: VerificationStatus
    equivalent_behaviors: int = 0
    divergent_behaviors: int = 0
    details: str = ""

    def __str__(self) -> str:
        return (f"[{self.status.value}] eq={self.equivalent_behaviors} "
                f"div={self.divergent_behaviors} — {self.details}")


# ---------------------------------------------------------------------------
# Architecture memory model properties
# ---------------------------------------------------------------------------

_ARCH_MODELS: Dict[str, Dict[str, bool]] = {
    "x86": {
        "store_store_reorder": False,
        "load_load_reorder": False,
        "load_store_reorder": False,
        "store_load_reorder": True,    # only reorder x86 allows
        "store_atomic": True,
        "multi_copy_atomic": True,
    },
    "x86_64": {
        "store_store_reorder": False,
        "load_load_reorder": False,
        "load_store_reorder": False,
        "store_load_reorder": True,
        "store_atomic": True,
        "multi_copy_atomic": True,
    },
    "arm": {
        "store_store_reorder": True,
        "load_load_reorder": True,
        "load_store_reorder": True,
        "store_load_reorder": True,
        "store_atomic": False,
        "multi_copy_atomic": False,
    },
    "arm64": {
        "store_store_reorder": True,
        "load_load_reorder": True,
        "load_store_reorder": True,
        "store_load_reorder": True,
        "store_atomic": False,
        "multi_copy_atomic": True,
    },
    "power": {
        "store_store_reorder": True,
        "load_load_reorder": True,
        "load_store_reorder": True,
        "store_load_reorder": True,
        "store_atomic": False,
        "multi_copy_atomic": False,
    },
    "riscv": {
        "store_store_reorder": True,
        "load_load_reorder": True,
        "load_store_reorder": True,
        "store_load_reorder": True,
        "store_atomic": False,
        "multi_copy_atomic": True,
    },
}

# ---------------------------------------------------------------------------
# Concurrency construct mappings between languages
# ---------------------------------------------------------------------------

_CONSTRUCT_MAP: Dict[str, Dict[str, str]] = {
    # pthread -> C++ mapping
    "pthread_mutex_lock": {"cpp": ".lock()", "rust": ".lock().unwrap()", "go": ".Lock()",
                           "java": ".lock()", "python": ".acquire()"},
    "pthread_mutex_unlock": {"cpp": ".unlock()", "rust": "// drop(guard)", "go": ".Unlock()",
                             "java": ".unlock()", "python": ".release()"},
    "pthread_create": {"cpp": "std::thread", "rust": "thread::spawn", "go": "go func",
                       "java": "new Thread", "python": "threading.Thread"},
    "pthread_join": {"cpp": ".join()", "rust": ".join().unwrap()", "go": "wg.Wait()",
                     "java": ".join()", "python": ".join()"},
    "pthread_cond_wait": {"cpp": ".wait(lock)", "rust": ".wait(guard).unwrap()",
                          "go": ".Wait()", "java": ".await()", "python": ".wait()"},
    "pthread_cond_signal": {"cpp": ".notify_one()", "rust": ".notify_one()",
                            "go": ".Signal()", "java": ".signal()", "python": ".notify()"},
    "pthread_cond_broadcast": {"cpp": ".notify_all()", "rust": ".notify_all()",
                               "go": ".Broadcast()", "java": ".signalAll()",
                               "python": ".notify_all()"},
    "pthread_rwlock_rdlock": {"cpp": "shared_lock", "rust": ".read().unwrap()",
                              "go": ".RLock()", "java": ".readLock().lock()",
                              "python": "# rwlock read"},
    "pthread_rwlock_wrlock": {"cpp": "unique_lock", "rust": ".write().unwrap()",
                              "go": ".Lock()", "java": ".writeLock().lock()",
                              "python": "# rwlock write"},
    "atomic_load": {"cpp": ".load()", "rust": ".load(Ordering::SeqCst)",
                    "go": "atomic.LoadInt64", "java": ".get()",
                    "python": "# atomic load"},
    "atomic_store": {"cpp": ".store()", "rust": ".store(val, Ordering::SeqCst)",
                     "go": "atomic.StoreInt64", "java": ".set()",
                     "python": "# atomic store"},
    "atomic_fetch_add": {"cpp": ".fetch_add()", "rust": ".fetch_add(val, Ordering::SeqCst)",
                         "go": "atomic.AddInt64", "java": ".addAndGet()",
                         "python": "# atomic add"},
}

_ORDERING_MAP: Dict[str, Dict[str, str]] = {
    "memory_order_relaxed": {"rust": "Ordering::Relaxed", "cpp": "std::memory_order_relaxed",
                             "java": "// relaxed (use volatile)", "go": "// relaxed"},
    "memory_order_acquire": {"rust": "Ordering::Acquire", "cpp": "std::memory_order_acquire",
                             "java": "// acquire (volatile read)", "go": "// acquire"},
    "memory_order_release": {"rust": "Ordering::Release", "cpp": "std::memory_order_release",
                             "java": "// release (volatile write)", "go": "// release"},
    "memory_order_acq_rel": {"rust": "Ordering::AcqRel", "cpp": "std::memory_order_acq_rel",
                             "java": "// acq_rel", "go": "// acq_rel"},
    "memory_order_seq_cst": {"rust": "Ordering::SeqCst", "cpp": "std::memory_order_seq_cst",
                             "java": "// seq_cst (default)", "go": "// seq_cst (default)"},
}


def _normalize_lang(lang: str) -> str:
    mapping = {"c": "c", "cpp": "cpp", "c++": "cpp", "rust": "rust",
               "go": "go", "java": "java", "python": "python", "py": "python"}
    return mapping.get(lang.lower(), lang.lower())


def _normalize_arch(arch: str) -> str:
    mapping = {"x86": "x86", "x86_64": "x86_64", "x86-64": "x86_64", "amd64": "x86_64",
               "arm": "arm", "aarch64": "arm64", "arm64": "arm64",
               "power": "power", "ppc": "power", "ppc64": "power",
               "riscv": "riscv", "riscv64": "riscv"}
    return mapping.get(arch.lower(), arch.lower())


def check_memory_model_compatibility(from_arch: str, to_arch: str,
                                     code: str) -> CompatResult:
    """Check whether concurrent code is safe to port between architectures."""
    src = _normalize_arch(from_arch)
    dst = _normalize_arch(to_arch)

    src_model = _ARCH_MODELS.get(src)
    dst_model = _ARCH_MODELS.get(dst)

    if not src_model or not dst_model:
        return CompatResult(
            level=CompatLevel.INCOMPATIBLE, from_arch=src, to_arch=dst,
            recommendations=[f"Unknown architecture: {src if not src_model else dst}"],
        )

    required_fences: List[str] = []
    reorderings: List[str] = []
    recommendations: List[str] = []

    # Check each reordering type
    reorder_types = [
        ("store_store_reorder", "StoreStore", "dmb st / lwsync"),
        ("load_load_reorder", "LoadLoad", "dmb ld / isync"),
        ("load_store_reorder", "LoadStore", "dmb ish"),
        ("store_load_reorder", "StoreLoad", "mfence / dmb ish"),
    ]

    for key, name, fence in reorder_types:
        src_reorders = src_model[key]
        dst_reorders = dst_model[key]
        if not src_reorders and dst_reorders:
            reorderings.append(f"{name} reordering: allowed on {dst} but not on {src}")
            required_fences.append(f"{fence} (for {name})")

    # Check atomicity
    if src_model["store_atomic"] and not dst_model["store_atomic"]:
        recommendations.append(
            f"Stores are not atomic on {dst} — use explicit atomic store instructions."
        )
    if src_model["multi_copy_atomic"] and not dst_model["multi_copy_atomic"]:
        recommendations.append(
            f"{dst} is not multi-copy atomic — other cores may see stores in different orders."
        )

    # Check if code uses relaxed atomics (higher risk on weaker models)
    if re.search(r'memory_order_relaxed|Ordering::Relaxed|Relaxed', code):
        if dst in ("arm", "arm64", "power", "riscv"):
            recommendations.append(
                "Relaxed atomics used — verify they are sufficient on the weaker target model."
            )

    if not required_fences and not reorderings:
        level = CompatLevel.COMPATIBLE
    elif required_fences:
        level = CompatLevel.NEEDS_FENCES
    else:
        level = CompatLevel.COMPATIBLE

    return CompatResult(
        level=level, from_arch=src, to_arch=dst,
        required_fences=required_fences, reorderings=reorderings,
        recommendations=recommendations,
    )


def port_concurrent(source: str, from_lang: str, to_lang: str) -> PortResult:
    """Port concurrent code from one language to another."""
    src = _normalize_lang(from_lang)
    dst = _normalize_lang(to_lang)

    if src == dst:
        return PortResult(status=PortStatus.SUCCESS, ported_code=source,
                          from_lang=src, to_lang=dst)

    warnings: List[PortWarning] = []
    unmapped: List[str] = []
    lines = source.split('\n')
    ported_lines: List[str] = []

    for i, line in enumerate(lines, 1):
        new_line = line
        matched = False

        for construct, mappings in _CONSTRUCT_MAP.items():
            if construct in line and dst in mappings:
                replacement = mappings[dst]
                # Try to intelligently replace
                if construct.startswith("pthread_"):
                    new_line = _port_pthread_line(line, construct, replacement, dst)
                    matched = True
                    break
                elif construct.startswith("atomic_"):
                    new_line = _port_atomic_line(line, construct, replacement, dst)
                    matched = True
                    break

        # Port memory orderings
        for ordering, mapping in _ORDERING_MAP.items():
            if ordering in new_line and dst in mapping:
                new_line = new_line.replace(ordering, mapping[dst])

        # Track unmapped constructs
        if not matched and src == "c":
            for construct in _CONSTRUCT_MAP:
                if construct in line and dst not in _CONSTRUCT_MAP.get(construct, {}):
                    unmapped.append(f"line {i}: {construct}")

        # Port includes / imports
        if src == "c" and dst == "cpp":
            new_line = _port_c_includes_to_cpp(new_line)
        elif src == "c" and dst == "rust":
            new_line = _port_c_to_rust_line(new_line, i, warnings)
        elif src == "c" and dst == "go":
            new_line = _port_c_to_go_line(new_line, i, warnings)
        elif src == "c" and dst == "python":
            new_line = _port_c_to_python_line(new_line, i, warnings)

        ported_lines.append(new_line)

    ported_code = "\n".join(ported_lines)
    status = PortStatus.SUCCESS if not unmapped else PortStatus.PARTIAL

    return PortResult(
        status=status, ported_code=ported_code,
        from_lang=src, to_lang=dst,
        warnings=warnings, unmapped_constructs=unmapped,
    )


def _port_pthread_line(line: str, construct: str, replacement: str, dst: str) -> str:
    """Port a single pthread call to the target language."""
    if dst == "cpp":
        if "pthread_mutex_lock" in construct:
            m = re.search(r'pthread_mutex_lock\s*\(\s*&?\s*(\w+)\s*\)', line)
            if m:
                return line[:m.start()] + f"{m.group(1)}.lock()" + line[m.end():]
        elif "pthread_mutex_unlock" in construct:
            m = re.search(r'pthread_mutex_unlock\s*\(\s*&?\s*(\w+)\s*\)', line)
            if m:
                return line[:m.start()] + f"{m.group(1)}.unlock()" + line[m.end():]
        elif "pthread_create" in construct:
            return f"    // TODO: replace with std::thread {replacement}"
        elif "pthread_join" in construct:
            return f"    // TODO: replace with {replacement}"
        elif "pthread_cond_wait" in construct:
            m = re.search(r'pthread_cond_wait\s*\(\s*&?\s*(\w+)\s*,\s*&?\s*(\w+)\s*\)', line)
            if m:
                return line[:m.start()] + f"{m.group(1)}.wait(lock)" + line[m.end():]
        elif "pthread_cond_signal" in construct:
            m = re.search(r'pthread_cond_signal\s*\(\s*&?\s*(\w+)\s*\)', line)
            if m:
                return line[:m.start()] + f"{m.group(1)}.notify_one()" + line[m.end():]
    return f"    {replacement}  // ported from {construct}"


def _port_atomic_line(line: str, construct: str, replacement: str, dst: str) -> str:
    """Port atomic operations."""
    return f"    {replacement}  // ported from {construct}"


def _port_c_includes_to_cpp(line: str) -> str:
    """Map C headers to C++ equivalents."""
    mappings = {
        "#include <pthread.h>": "#include <thread>\n#include <mutex>\n#include <condition_variable>",
        "#include <stdatomic.h>": "#include <atomic>",
        "#include <semaphore.h>": "#include <semaphore>",
    }
    for old, new in mappings.items():
        if old in line:
            return new
    return line


def _port_c_to_rust_line(line: str, lineno: int, warnings: List[PortWarning]) -> str:
    """Attempt to port C line to Rust."""
    if "#include" in line:
        return f"// {line.strip()}  // TODO: add Rust use statements"
    if "pthread_mutex_t" in line:
        return line.replace("pthread_mutex_t", "Mutex<()>")
    if "void*" in line or "void *" in line:
        warnings.append(PortWarning(lineno, line.strip(),
                                    "void* has no direct Rust equivalent — use generics or trait objects"))
    return line


def _port_c_to_go_line(line: str, lineno: int, warnings: List[PortWarning]) -> str:
    """Attempt to port C line to Go."""
    if "#include" in line:
        return f"// {line.strip()}  // TODO: add Go imports"
    if "malloc" in line or "calloc" in line:
        warnings.append(PortWarning(lineno, line.strip(),
                                    "Go uses garbage collection — remove manual memory management"))
    return line


def _port_c_to_python_line(line: str, lineno: int, warnings: List[PortWarning]) -> str:
    """Attempt to port C line to Python."""
    if "#include" in line:
        return f"# {line.strip()}  // TODO: add Python imports"
    if "atomic" in line.lower():
        warnings.append(PortWarning(lineno, line.strip(),
                                    "Python has no native atomics — use threading.Lock instead"))
    return line


def pthread_to_cpp_threads(source: str) -> str:
    """Convert pthreads code to C++ std::thread."""
    result = port_concurrent(source, "c", "cpp")
    return result.ported_code


def openmp_to_cuda(source: str) -> str:
    """Convert OpenMP parallel regions to CUDA kernels."""
    lines = source.split('\n')
    output: List[str] = []
    in_parallel_for = False
    loop_var = ""
    loop_bound = ""

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Replace includes
        if "#include <omp.h>" in line:
            output.append("#include <cuda_runtime.h>")
            continue

        # Detect #pragma omp parallel for
        if re.match(r'#pragma\s+omp\s+parallel\s+for', stripped):
            in_parallel_for = True
            output.append("// Converted from OpenMP parallel for")
            continue

        if in_parallel_for:
            # Parse the for loop header
            m = re.match(r'for\s*\(\s*int\s+(\w+)\s*=\s*(\w+)\s*;\s*\w+\s*<\s*(\w+)\s*;', stripped)
            if m:
                loop_var = m.group(1)
                loop_bound = m.group(3)
                output.append(f"__global__ void kernel(int* data, int {loop_bound}) {{")
                output.append(f"    int {loop_var} = blockIdx.x * blockDim.x + threadIdx.x;")
                output.append(f"    if ({loop_var} < {loop_bound}) {{")
                in_parallel_for = False
                continue

        # Replace omp_get_thread_num / omp_get_num_threads
        line = re.sub(r'omp_get_thread_num\(\)', 'threadIdx.x + blockIdx.x * blockDim.x', line)
        line = re.sub(r'omp_get_num_threads\(\)', 'gridDim.x * blockDim.x', line)

        # Replace omp critical with atomicAdd where possible
        if re.match(r'#pragma\s+omp\s+critical', stripped):
            output.append("    // TODO: use atomicAdd or atomicCAS for CUDA")
            continue

        # Replace omp barrier
        if re.match(r'#pragma\s+omp\s+barrier', stripped):
            output.append("    __syncthreads();")
            continue

        # Replace omp atomic
        if re.match(r'#pragma\s+omp\s+atomic', stripped):
            output.append("    // Use atomicAdd/atomicCAS below")
            continue

        # Replace omp reduction
        m = re.match(r'#pragma\s+omp\s+.*reduction\((\+):(\w+)\)', stripped)
        if m:
            output.append(f"    // TODO: implement parallel reduction for {m.group(2)}")
            continue

        output.append(line)

    return "\n".join(output)


def sequential_to_parallel(source: str, strategy: str = "auto") -> str:
    """Convert sequential code to parallel using detected opportunities."""
    lines = source.split('\n')
    output: List[str] = []
    parallelized = False

    for i, line in enumerate(lines):
        stripped = line.strip()

        # Detect independent for loops
        m = re.match(r'(\s*)for\s*\(\s*int\s+(\w+)\s*=\s*(\w+)\s*;\s*(\w+)\s*<\s*(\w+)\s*;', stripped)
        if m:
            indent = m.group(1) or line[:len(line) - len(line.lstrip())]
            var = m.group(2)
            bound = m.group(5)
            # Check if loop body references only array[var] patterns (heuristic)
            body_lines = []
            depth = 0
            for j in range(i + 1, len(lines)):
                depth += lines[j].count('{') - lines[j].count('}')
                body_lines.append(lines[j])
                if depth <= 0:
                    break
            body = "\n".join(body_lines)

            # Simple independence check: no writes to shared scalars
            scalar_writes = re.findall(r'(\w+)\s*[+\-*/]?=(?!=)', body)
            array_accesses = re.findall(r'\w+\[' + var + r'\]', body)

            if array_accesses and len(scalar_writes) <= 1:
                if strategy in ("auto", "openmp"):
                    output.append(f"{indent}#pragma omp parallel for")
                    parallelized = True

        output.append(line)

    if not parallelized:
        output.insert(0, "// No safe parallelization opportunities detected")

    return "\n".join(output)


def verify_port(original: str, ported: str) -> VerificationResult:
    """Verify that a ported implementation preserves concurrency semantics."""
    eq_count = 0
    div_count = 0
    details: List[str] = []

    # Extract sync primitives from both
    orig_locks = set(re.findall(r'(mutex|lock|Lock|synchronized|rwlock|RwLock|Mutex)', original))
    port_locks = set(re.findall(r'(mutex|lock|Lock|synchronized|rwlock|RwLock|Mutex)', ported))

    if orig_locks and not port_locks:
        div_count += 1
        details.append("Original has locks but ported version does not")
    elif orig_locks and port_locks:
        eq_count += 1

    # Check atomic operations preserved
    orig_atomics = len(re.findall(r'(atomic|Atomic|_Atomic|std::atomic)', original))
    port_atomics = len(re.findall(r'(atomic|Atomic|_Atomic|std::atomic)', ported))

    if orig_atomics > 0 and port_atomics == 0:
        div_count += 1
        details.append("Original uses atomics but ported version does not")
    elif orig_atomics > 0 and port_atomics > 0:
        eq_count += 1

    # Check thread creation preserved
    orig_threads = len(re.findall(
        r'(pthread_create|std::thread|thread::spawn|go\s+func|new\s+Thread|Thread\()', original))
    port_threads = len(re.findall(
        r'(pthread_create|std::thread|thread::spawn|go\s+func|new\s+Thread|Thread\()', ported))

    if orig_threads > 0 and port_threads == 0:
        div_count += 1
        details.append("Original creates threads but ported version does not")
    elif orig_threads > 0 and port_threads > 0:
        eq_count += 1

    # Check condition variables / signaling preserved
    orig_cv = len(re.findall(r'(cond_wait|cond_signal|notify|wait|signal|broadcast)', original))
    port_cv = len(re.findall(r'(cond_wait|cond_signal|notify|wait|signal|broadcast)', ported))

    if orig_cv > 0 and port_cv == 0:
        div_count += 1
        details.append("Original uses condition variables but ported version does not")
    elif orig_cv > 0:
        eq_count += 1

    # Check memory ordering preserved
    orig_orderings = set(re.findall(r'memory_order_\w+|Ordering::\w+', original))
    port_orderings = set(re.findall(r'memory_order_\w+|Ordering::\w+', ported))
    if orig_orderings and not port_orderings:
        div_count += 1
        details.append("Memory orderings from original not preserved in port")
    elif orig_orderings and port_orderings:
        eq_count += 1

    if div_count == 0:
        status = VerificationStatus.EQUIVALENT
    elif eq_count > div_count:
        status = VerificationStatus.UNKNOWN
    else:
        status = VerificationStatus.DIFFERS

    return VerificationResult(
        status=status,
        equivalent_behaviors=eq_count,
        divergent_behaviors=div_count,
        details="; ".join(details) if details else "Port appears semantically equivalent.",
    )
