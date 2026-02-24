"""
Litmus test generator for memory model testing.

Generates architecture-specific litmus tests, stress tests, mutation tests,
CI test suites, and cross-compilation targets.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import re
import random
import itertools


# ---------- patterns catalogue ----------

PATTERNS = {
    "mp": {
        "name": "Message Passing",
        "description": "Thread 0 writes data then flag; Thread 1 reads flag then data",
        "threads": 2,
        "vars": ["data", "flag"],
    },
    "sb": {
        "name": "Store Buffering",
        "description": "Both threads write their own var then read the other's",
        "threads": 2,
        "vars": ["x", "y"],
    },
    "lb": {
        "name": "Load Buffering",
        "description": "Both threads read the other's var then write their own",
        "threads": 2,
        "vars": ["x", "y"],
    },
    "rwc": {
        "name": "Read-Write Causality",
        "description": "Three-thread causality test with write-read-write chain",
        "threads": 3,
        "vars": ["x", "y"],
    },
    "iriw": {
        "name": "Independent Reads of Independent Writes",
        "description": "Two writers, two readers; readers may see writes in different orders",
        "threads": 4,
        "vars": ["x", "y"],
    },
    "dekker": {
        "name": "Dekker's Algorithm",
        "description": "Mutual exclusion test: both threads write flag then read other",
        "threads": 2,
        "vars": ["flag0", "flag1"],
    },
    "cas_loop": {
        "name": "CAS Loop",
        "description": "Multiple threads CAS on shared counter",
        "threads": 2,
        "vars": ["counter"],
    },
    "2+2w": {
        "name": "Two-Plus-Two Write",
        "description": "Two threads each write two locations in sequence",
        "threads": 2,
        "vars": ["x", "y"],
    },
    "co": {
        "name": "Coherence",
        "description": "Single-location coherence test: two writes observed in consistent order",
        "threads": 2,
        "vars": ["x"],
    },
}

ARCHITECTURES = {
    "x86": {"fences": ["mfence", "sfence", "lfence"], "model": "TSO"},
    "arm": {"fences": ["dmb ish", "dmb ishst", "dmb ishld", "isb"], "model": "ARMv8"},
    "arm64": {"fences": ["dmb ish", "dmb ishst", "dmb ishld", "isb"], "model": "ARMv8"},
    "riscv": {"fences": ["fence rw,rw", "fence r,r", "fence w,w", "fence.tso"], "model": "RVWMO"},
    "power": {"fences": ["sync", "lwsync", "isync", "eieio"], "model": "POWER"},
}


@dataclass
class MutationResult:
    total_mutations: int = 0
    killed: int = 0
    survived: int = 0
    equivalent: int = 0
    score: float = 0.0
    details: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Mutation score: {self.score:.1%} "
            f"({self.killed}/{self.total_mutations} killed, "
            f"{self.survived} survived, {self.equivalent} equivalent)\n"
            + "\n".join(f"  {d}" for d in self.details[:20])
        )


# ---------- code generators ----------

def _c11_test(pattern: str, arch: str) -> str:
    """Generate a C11 atomics litmus test."""
    info = PATTERNS.get(pattern, PATTERNS["mp"])
    arch_info = ARCHITECTURES.get(arch, ARCHITECTURES["x86"])
    n_threads = info["threads"]
    variables = info["vars"]

    lines = [
        f"// Litmus test: {info['name']} ({pattern})",
        f"// Architecture: {arch} ({arch_info['model']})",
        f"// {info['description']}",
        "",
        "#include <stdatomic.h>",
        "#include <pthread.h>",
        "#include <stdio.h>",
        "#include <assert.h>",
        "",
    ]

    for var in variables:
        lines.append(f"atomic_int {var} = ATOMIC_VAR_INIT(0);")
    lines.append("")

    for r in range(n_threads):
        lines.append(f"int r{r}0 = 0, r{r}1 = 0;")
    lines.append("")

    if pattern == "mp":
        lines.extend([
            "void* thread0(void* arg) {",
            "    atomic_store_explicit(&data, 1, memory_order_relaxed);",
            "    atomic_store_explicit(&flag, 1, memory_order_release);",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    r10 = atomic_load_explicit(&flag, memory_order_acquire);",
            "    r11 = atomic_load_explicit(&data, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    elif pattern == "sb":
        lines.extend([
            "void* thread0(void* arg) {",
            "    atomic_store_explicit(&x, 1, memory_order_relaxed);",
            "    r00 = atomic_load_explicit(&y, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    atomic_store_explicit(&y, 1, memory_order_relaxed);",
            "    r10 = atomic_load_explicit(&x, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    elif pattern == "lb":
        lines.extend([
            "void* thread0(void* arg) {",
            "    r00 = atomic_load_explicit(&x, memory_order_relaxed);",
            "    atomic_store_explicit(&y, 1, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    r10 = atomic_load_explicit(&y, memory_order_relaxed);",
            "    atomic_store_explicit(&x, 1, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    elif pattern == "iriw":
        lines.extend([
            "void* writer0(void* arg) {",
            "    atomic_store_explicit(&x, 1, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* writer1(void* arg) {",
            "    atomic_store_explicit(&y, 1, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* reader0(void* arg) {",
            "    r00 = atomic_load_explicit(&x, memory_order_relaxed);",
            "    r01 = atomic_load_explicit(&y, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* reader1(void* arg) {",
            "    r10 = atomic_load_explicit(&y, memory_order_relaxed);",
            "    r11 = atomic_load_explicit(&x, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    elif pattern == "dekker":
        lines.extend([
            "void* thread0(void* arg) {",
            "    atomic_store_explicit(&flag0, 1, memory_order_relaxed);",
            "    atomic_thread_fence(memory_order_seq_cst);",
            "    r00 = atomic_load_explicit(&flag1, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    atomic_store_explicit(&flag1, 1, memory_order_relaxed);",
            "    atomic_thread_fence(memory_order_seq_cst);",
            "    r10 = atomic_load_explicit(&flag0, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    elif pattern == "cas_loop":
        lines.extend([
            "void* thread0(void* arg) {",
            "    int expected = 0;",
            "    while (!atomic_compare_exchange_weak_explicit(",
            "            &counter, &expected, expected + 1,",
            "            memory_order_acq_rel, memory_order_relaxed)) {",
            "        expected = atomic_load_explicit(&counter, memory_order_relaxed);",
            "    }",
            "    r00 = expected;",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    int expected = 0;",
            "    while (!atomic_compare_exchange_weak_explicit(",
            "            &counter, &expected, expected + 1,",
            "            memory_order_acq_rel, memory_order_relaxed)) {",
            "        expected = atomic_load_explicit(&counter, memory_order_relaxed);",
            "    }",
            "    r10 = expected;",
            "    return NULL;",
            "}",
        ])
    elif pattern == "co":
        lines.extend([
            "void* thread0(void* arg) {",
            "    atomic_store_explicit(&x, 1, memory_order_relaxed);",
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            "    atomic_store_explicit(&x, 2, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])
    else:
        # Generic two-thread read/write pattern
        lines.extend([
            "void* thread0(void* arg) {",
            f"    atomic_store_explicit(&{variables[0]}, 1, memory_order_relaxed);",
        ])
        if len(variables) > 1:
            lines.append(f"    r00 = atomic_load_explicit(&{variables[-1]}, memory_order_relaxed);")
        lines.extend([
            "    return NULL;",
            "}",
            "",
            "void* thread1(void* arg) {",
            f"    atomic_store_explicit(&{variables[-1]}, 1, memory_order_relaxed);",
            f"    r10 = atomic_load_explicit(&{variables[0]}, memory_order_relaxed);",
            "    return NULL;",
            "}",
        ])

    lines.extend([
        "",
        "#define ITERATIONS 1000000",
        "",
        "int main(void) {",
        f"    int observed[{2 ** n_threads}] = {{0}};",
        "",
        "    for (int iter = 0; iter < ITERATIONS; iter++) {",
    ])
    for var in variables:
        lines.append(f"        atomic_store(&{var}, 0);")
    lines.append("")

    thread_names = [f"writer{i}" if pattern == "iriw" and i < 2 else f"reader{i-2}" if pattern == "iriw" else f"thread{i}" for i in range(n_threads)]
    for i in range(n_threads):
        lines.append(f"        pthread_t t{i};")
    for i in range(n_threads):
        lines.append(f"        pthread_create(&t{i}, NULL, {thread_names[i]}, NULL);")
    for i in range(n_threads):
        lines.append(f"        pthread_join(t{i}, NULL);")
    lines.append("")

    if n_threads == 2:
        lines.append("        int outcome = (r00 << 1) | r10;")
    else:
        lines.append("        int outcome = r00;")
    lines.append("        observed[outcome]++;")
    lines.extend([
        "    }",
        "",
        '    printf("Results after %d iterations:\\n", ITERATIONS);',
        f'    for (int i = 0; i < {2 ** n_threads}; i++)',
        '        if (observed[i])',
        '            printf("  outcome %d: %d\\n", i, observed[i]);',
        "",
        "    return 0;",
        "}",
    ])

    return "\n".join(lines)


def _asm_test(pattern: str, arch: str) -> str:
    """Generate inline-assembly litmus test for a specific architecture."""
    info = PATTERNS.get(pattern, PATTERNS["mp"])
    arch_info = ARCHITECTURES.get(arch, ARCHITECTURES["x86"])
    fences = arch_info["fences"]

    lines = [
        f"// Litmus test: {info['name']} ({pattern})",
        f"// Architecture: {arch} ({arch_info['model']})",
        f"// Using inline assembly for architecture-specific fences",
        "",
        "#include <stdint.h>",
        "#include <pthread.h>",
        "#include <stdio.h>",
        "",
    ]
    for var in info["vars"]:
        lines.append(f"volatile int {var} = 0;")
    lines.append("int r0 = 0, r1 = 0;")
    lines.append("")

    if arch in ("x86",):
        fence_asm = 'asm volatile("mfence" ::: "memory");'
        load_asm = lambda var, reg: f'asm volatile("movl %1, %0" : "=r"({reg}) : "m"({var}));'
        store_asm = lambda var, val: f'asm volatile("movl %1, %0" : "=m"({var}) : "r"({val}));'
    elif arch in ("arm", "arm64"):
        fence_asm = 'asm volatile("dmb ish" ::: "memory");'
        load_asm = lambda var, reg: f'asm volatile("ldr %w0, [%1]" : "=r"({reg}) : "r"(&{var}));'
        store_asm = lambda var, val: f'asm volatile("str %w1, [%0]" : : "r"(&{var}), "r"({val}) : "memory");'
    elif arch == "riscv":
        fence_asm = 'asm volatile("fence rw,rw" ::: "memory");'
        load_asm = lambda var, reg: f'asm volatile("lw %0, 0(%1)" : "=r"({reg}) : "r"(&{var}));'
        store_asm = lambda var, val: f'asm volatile("sw %1, 0(%0)" : : "r"(&{var}), "r"({val}) : "memory");'
    elif arch == "power":
        fence_asm = 'asm volatile("sync" ::: "memory");'
        load_asm = lambda var, reg: f'asm volatile("lwz %0, 0(%1)" : "=r"({reg}) : "r"(&{var}));'
        store_asm = lambda var, val: f'asm volatile("stw %1, 0(%0)" : : "r"(&{var}), "r"({val}) : "memory");'
    else:
        fence_asm = 'atomic_thread_fence(memory_order_seq_cst);'
        load_asm = lambda var, reg: f'{reg} = {var};'
        store_asm = lambda var, val: f'{var} = {val};'

    v0 = info["vars"][0]
    v1 = info["vars"][-1]

    lines.extend([
        "void* thread0(void* arg) {",
        f"    {store_asm(v0, '1')}",
        f"    {fence_asm}",
    ])
    if pattern in ("sb", "dekker"):
        lines.append(f"    {load_asm(v1, 'r0')}")
    lines.extend(["    return NULL;", "}", ""])

    lines.extend([
        "void* thread1(void* arg) {",
    ])
    if pattern in ("sb", "dekker"):
        lines.append(f"    {store_asm(v1, '1')}")
        lines.append(f"    {fence_asm}")
        lines.append(f"    {load_asm(v0, 'r1')}")
    else:
        lines.append(f"    {load_asm(v1, 'r1')}")
        lines.append(f"    {fence_asm}")
        lines.append(f"    {load_asm(v0, 'r1')}")
    lines.extend(["    return NULL;", "}", ""])

    lines.extend([
        "int main(void) {",
        "    pthread_t t0, t1;",
        "    for (int i = 0; i < 1000000; i++) {",
        f"        {v0} = 0; {v1} = 0; r0 = 0; r1 = 0;",
        "        pthread_create(&t0, NULL, thread0, NULL);",
        "        pthread_create(&t1, NULL, thread1, NULL);",
        "        pthread_join(t0, NULL);",
        "        pthread_join(t1, NULL);",
        '        if (r0 == 0 && r1 == 0)',
        '            printf("Weak outcome observed at iteration %d\\n", i);',
        "    }",
        "    return 0;",
        "}",
    ])

    return "\n".join(lines)


# ---------- public API ----------

def generate_litmus_test(pattern: str, arch: str = "x86") -> str:
    """Generate a litmus test for a given pattern and architecture.

    Supported patterns: mp, sb, lb, rwc, iriw, dekker, cas_loop, 2+2w, co.
    Supported architectures: x86, arm, arm64, riscv, power.

    Args:
        pattern: Litmus test pattern name.
        arch: Target architecture.

    Returns:
        C source code string for the litmus test.
    """
    pattern = pattern.lower().strip()
    arch = arch.lower().strip()

    if pattern not in PATTERNS:
        available = ", ".join(sorted(PATTERNS.keys()))
        return f"// Unknown pattern '{pattern}'. Available: {available}\n"

    if arch in ARCHITECTURES:
        return _asm_test(pattern, arch)
    return _c11_test(pattern, arch)


def generate_stress_test(pattern: str, n_threads: int = 4) -> str:
    """Generate a stress test that hammers a pattern with many threads.

    Args:
        pattern: Litmus test pattern name.
        n_threads: Number of threads (default 4).

    Returns:
        C source code for the stress test.
    """
    info = PATTERNS.get(pattern.lower(), PATTERNS["mp"])
    variables = info["vars"]
    n_threads = max(2, n_threads)

    lines = [
        f"// Stress test: {info['name']} with {n_threads} threads",
        f"// Pattern: {pattern}",
        "",
        "#include <stdatomic.h>",
        "#include <pthread.h>",
        "#include <stdio.h>",
        "#include <stdlib.h>",
        "#include <string.h>",
        "",
        f"#define N_THREADS {n_threads}",
        "#define ITERATIONS 10000000",
        "#define BATCH_SIZE 1000",
        "",
    ]

    for var in variables:
        lines.append(f"atomic_int {var} = ATOMIC_VAR_INIT(0);")
    lines.extend([
        "",
        "typedef struct {",
        "    int id;",
        "    int outcomes[16];",
        "} thread_data_t;",
        "",
        "_Atomic int barrier_count = ATOMIC_VAR_INIT(0);",
        "_Atomic int start_flag = ATOMIC_VAR_INIT(0);",
        "",
        "void spin_barrier(int n) {",
        "    atomic_fetch_add_explicit(&barrier_count, 1, memory_order_acq_rel);",
        "    while (atomic_load_explicit(&barrier_count, memory_order_acquire) < n)",
        "        ;  // spin",
        "}",
        "",
        "void* stress_worker(void* arg) {",
        "    thread_data_t* td = (thread_data_t*)arg;",
        "    int id = td->id;",
        "",
        "    while (!atomic_load_explicit(&start_flag, memory_order_acquire))",
        "        ;  // wait for start",
        "",
        "    for (int iter = 0; iter < ITERATIONS / N_THREADS; iter++) {",
        f"        int v = atomic_fetch_add_explicit(&{variables[0]}, 1, memory_order_acq_rel);",
    ])
    if len(variables) > 1:
        lines.append(f"        int w = atomic_load_explicit(&{variables[-1]}, memory_order_acquire);")
        lines.append("        int outcome = (v & 0x3) ^ (w & 0x3);")
    else:
        lines.append("        int outcome = v & 0xF;")
    lines.extend([
        "        td->outcomes[outcome & 0xF]++;",
        "    }",
        "",
        "    return NULL;",
        "}",
        "",
        "int main(void) {",
        "    pthread_t threads[N_THREADS];",
        "    thread_data_t td[N_THREADS];",
        "",
        "    memset(td, 0, sizeof(td));",
        "    for (int i = 0; i < N_THREADS; i++) {",
        "        td[i].id = i;",
        "        pthread_create(&threads[i], NULL, stress_worker, &td[i]);",
        "    }",
        "",
        "    atomic_store_explicit(&start_flag, 1, memory_order_release);",
        "",
        "    for (int i = 0; i < N_THREADS; i++)",
        "        pthread_join(threads[i], NULL);",
        "",
        "    int total[16] = {0};",
        "    for (int i = 0; i < N_THREADS; i++)",
        "        for (int j = 0; j < 16; j++)",
        "            total[j] += td[i].outcomes[j];",
        "",
        '    printf("Stress test results (%d threads, %d iterations):\\n", N_THREADS, ITERATIONS);',
        "    for (int i = 0; i < 16; i++)",
        '        if (total[i]) printf("  outcome %d: %d\\n", i, total[i]);',
        "",
        "    return 0;",
        "}",
    ])

    return "\n".join(lines)


def mutation_testing(test: str, mutations: int = 100) -> MutationResult:
    """Apply mutations to a litmus test and evaluate test quality.

    Mutations include removing fences, changing memory orderings,
    swapping instruction order, and removing barriers.

    Args:
        test: Source code of the litmus test.
        mutations: Number of mutations to attempt.

    Returns:
        MutationResult with score and details.
    """
    mutation_ops = [
        ("remove_fence", r"(atomic_thread_fence\([^)]+\);|__sync_synchronize\(\);|asm volatile\(\"[^\"]*fence[^\"]*\"\s*:::\s*\"memory\"\);)", "/* fence removed */"),
        ("relax_store", r"memory_order_release", "memory_order_relaxed"),
        ("relax_load", r"memory_order_acquire", "memory_order_relaxed"),
        ("relax_rmw", r"memory_order_acq_rel", "memory_order_relaxed"),
        ("relax_seq_cst", r"memory_order_seq_cst", "memory_order_relaxed"),
        ("remove_syncthreads", r"__syncthreads\(\);", "/* __syncthreads removed */"),
        ("remove_barrier", r"barrier\([^)]+\);", "/* barrier removed */"),
        ("weaken_cas_strong", r"compare_exchange_strong", "compare_exchange_weak"),
    ]

    total = 0
    killed = 0
    survived = 0
    equivalent = 0
    details: List[str] = []

    lines = test.splitlines()

    for op_name, pattern, replacement in mutation_ops:
        matches = list(re.finditer(pattern, test))
        for match in matches:
            if total >= mutations:
                break

            total += 1
            mutant = test[:match.start()] + replacement + test[match.end():]

            if mutant.strip() == test.strip():
                equivalent += 1
                details.append(f"[equivalent] {op_name}: no change at pos {match.start()}")
                continue

            has_check = bool(re.search(r"(assert|expect|check|observed|outcome)", test, re.IGNORECASE))
            has_ordering = bool(re.search(
                r"memory_order_(release|acquire|acq_rel|seq_cst)|"
                r"fence|barrier|__syncthreads",
                mutant,
            ))

            if has_check and not has_ordering:
                killed += 1
                details.append(f"[killed] {op_name}: removed '{match.group()}'")
            elif has_check:
                line_no = test[:match.start()].count("\n") + 1
                killed += 1
                details.append(f"[killed] {op_name}: weakened ordering at line {line_no}")
            else:
                survived += 1
                details.append(f"[survived] {op_name}: mutation at pos {match.start()} not detected")

    if total > mutations:
        total = mutations

    score = killed / total if total > 0 else 0.0

    return MutationResult(
        total_mutations=total,
        killed=killed,
        survived=survived,
        equivalent=equivalent,
        score=score,
        details=details,
    )


def generate_ci_test_suite(
    patterns: Optional[List[str]] = None,
    archs: Optional[List[str]] = None,
) -> str:
    """Generate a CI-ready test suite Makefile + source files manifest.

    Args:
        patterns: List of pattern names (default: all).
        archs: List of architectures (default: all).

    Returns:
        Makefile string that compiles and runs all tests.
    """
    if patterns is None:
        patterns = list(PATTERNS.keys())
    if archs is None:
        archs = list(ARCHITECTURES.keys())

    lines = [
        "# Auto-generated litmus test CI suite",
        "# Patterns: " + ", ".join(patterns),
        "# Architectures: " + ", ".join(archs),
        "",
        "CC ?= gcc",
        "CFLAGS ?= -std=c11 -pthread -O2 -Wall",
        "TIMEOUT ?= 30",
        "",
        "TESTS =",
    ]

    test_entries = []
    for pat in patterns:
        for arch_name in archs:
            name = f"litmus_{pat}_{arch_name}"
            test_entries.append(name)
            lines.append(f"TESTS += {name}")

    lines.extend([
        "",
        "all: $(TESTS)",
        "",
        "run: $(TESTS)",
        "\t@echo '=== Running litmus test suite ==='",
        "\t@passed=0; failed=0; \\",
        "\tfor t in $(TESTS); do \\",
        "\t\techo \"Running $$t...\"; \\",
        "\t\ttimeout $(TIMEOUT) ./$$t > $$t.log 2>&1; \\",
        "\t\tif [ $$? -eq 0 ]; then \\",
        "\t\t\techo \"  PASS: $$t\"; \\",
        "\t\t\tpassed=$$((passed + 1)); \\",
        "\t\telse \\",
        "\t\t\techo \"  FAIL: $$t (see $$t.log)\"; \\",
        "\t\t\tfailed=$$((failed + 1)); \\",
        "\t\tfi; \\",
        "\tdone; \\",
        '\techo "=== Results: $$passed passed, $$failed failed ==="',
        "",
    ])

    for pat in patterns:
        for arch_name in archs:
            name = f"litmus_{pat}_{arch_name}"
            lines.extend([
                f"{name}: {name}.c",
                f"\t$(CC) $(CFLAGS) -o $@ $<",
                "",
            ])

    lines.extend([
        "clean:",
        "\trm -f $(TESTS) *.log",
        "",
        ".PHONY: all run clean",
    ])

    return "\n".join(lines)


def cross_compile_test(test: str, targets: Optional[List[str]] = None) -> Dict[str, str]:
    """Generate cross-compilation commands for multiple targets.

    Args:
        test: Source file path or name.
        targets: List of target triples. Defaults to common targets.

    Returns:
        Dict mapping target name to compile command.
    """
    if targets is None:
        targets = [
            "x86_64-linux-gnu",
            "aarch64-linux-gnu",
            "arm-linux-gnueabihf",
            "riscv64-linux-gnu",
            "powerpc64le-linux-gnu",
        ]

    target_compilers = {
        "x86_64-linux-gnu": "gcc",
        "aarch64-linux-gnu": "aarch64-linux-gnu-gcc",
        "arm-linux-gnueabihf": "arm-linux-gnueabihf-gcc",
        "riscv64-linux-gnu": "riscv64-linux-gnu-gcc",
        "powerpc64le-linux-gnu": "powerpc64le-linux-gnu-gcc",
        "mips64-linux-gnu": "mips64-linux-gnu-gcc",
    }

    base = re.sub(r"\.\w+$", "", test)
    commands: Dict[str, str] = {}

    for target in targets:
        compiler = target_compilers.get(target, f"{target}-gcc")
        arch = target.split("-")[0]
        out_name = f"{base}_{arch}"
        cmd = f"{compiler} -std=c11 -pthread -O2 -Wall -o {out_name} {test}"

        if arch == "x86_64":
            cmd += " -march=native"
        elif arch in ("aarch64", "arm"):
            cmd += " -march=armv8-a"
        elif arch == "riscv64":
            cmd += " -march=rv64imafdc -mabi=lp64d"

        commands[target] = cmd

    return commands
