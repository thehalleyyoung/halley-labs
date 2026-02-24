"""Hardware testing infrastructure for litmus tests on real hardware."""

import collections
import json
import os
import platform
import random
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Set, Tuple


class TestStatus(Enum):
    PASS = auto()
    FAIL = auto()
    WEAK = auto()
    ERROR = auto()
    TIMEOUT = auto()
    SKIPPED = auto()

    def __str__(self) -> str:
        return self.name


class ArchFamily(Enum):
    X86 = auto()
    ARM = auto()
    AARCH64 = auto()
    RISCV = auto()
    POWER = auto()
    SPARC = auto()
    MIPS = auto()

    def __str__(self) -> str:
        return self.name


class CacheLevel(Enum):
    L1 = auto()
    L2 = auto()
    L3 = auto()
    LLC = auto()

    def __str__(self) -> str:
        return self.name


@dataclass
class HardwareInfo:
    arch: ArchFamily
    model_name: str
    cores: int
    threads_per_core: int
    cache_sizes: Dict[str, int]
    cache_line_size: int
    numa_nodes: int
    frequency_ghz: float

    def __str__(self) -> str:
        return (f"{self.model_name} ({self.arch}, {self.cores}C/"
                f"{self.cores * self.threads_per_core}T, "
                f"{self.frequency_ghz:.2f} GHz, {self.numa_nodes} NUMA)")


@dataclass
class LitmusTestDef:
    name: str
    threads: int
    c_source: str
    expected_behaviors: Dict[str, bool]
    description: str

    def __str__(self) -> str:
        weak = [k for k, v in self.expected_behaviors.items() if not v]
        return f"LitmusTest({self.name}, {self.threads}T, weak={weak})"


@dataclass
class HardwareResult:
    test: LitmusTestDef
    iterations: int
    outcomes: Dict[str, int]
    weak_behaviors_found: List[str]
    weak_behavior_count: int
    total_time_s: float
    status: TestStatus

    def __str__(self) -> str:
        rate = (self.weak_behavior_count / max(self.iterations, 1)) * 100
        return (f"Result({self.test.name}: {self.status}, "
                f"weak={self.weak_behavior_count}/{self.iterations} "
                f"({rate:.4f}%), {self.total_time_s:.2f}s)")


@dataclass
class WeakBehaviorReport:
    tests: List[HardwareResult]
    total_tests: int
    tests_with_weak: int
    total_weak_observations: int
    architecture: str
    hardware_info: HardwareInfo

    def __str__(self) -> str:
        return (f"WeakReport({self.architecture}: {self.tests_with_weak}/"
                f"{self.total_tests} tests showed weak, "
                f"{self.total_weak_observations} total observations)")


@dataclass
class StressResult:
    test: LitmusTestDef
    duration_s: float
    total_iterations: int
    outcomes: Dict[str, int]
    weak_rate: float
    peak_weak_rate: float
    convergence_history: List[Tuple[int, float]]

    def __str__(self) -> str:
        return (f"Stress({self.test.name}: {self.duration_s:.1f}s, "
                f"rate={self.weak_rate:.6f}, peak={self.peak_weak_rate:.6f})")


@dataclass
class ArchFingerprint:
    hardware_info: HardwareInfo
    observed_behaviors: Dict[str, Set[str]]
    matches_model: str
    confidence: float
    anomalies: List[str]

    def __str__(self) -> str:
        return (f"Fingerprint({self.matches_model}, "
                f"confidence={self.confidence:.1%}, "
                f"anomalies={len(self.anomalies)})")


@dataclass
class ImplComparison:
    implementations: List[str]
    results: Dict[str, Dict[str, HardwareResult]]
    fastest: str
    most_correct: str
    summary: str

    def __str__(self) -> str:
        return (f"Comparison({len(self.implementations)} impls, "
                f"fastest={self.fastest}, best={self.most_correct})")


# ---------------------------------------------------------------------------
# Built-in litmus test C source templates
# ---------------------------------------------------------------------------

_MP_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1;
pthread_barrier_t bar;

void *thread0(void *arg) {
    for (int i = 0; i < *(int*)arg; i++) {
        x = 0; y = 0; r0 = 0; r1 = 0;
        pthread_barrier_wait(&bar);
        x = 1; y = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread1(void *arg) {
    int counts[4] = {0};
    for (int i = 0; i < *(int*)arg; i++) {
        pthread_barrier_wait(&bar);
        r0 = y; r1 = x;
        pthread_barrier_wait(&bar);
        counts[r0 * 2 + r1]++;
    }
    printf("0,0:%d\n0,1:%d\n1,0:%d\n1,1:%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 2);
    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread0, &n);
    pthread_create(&t1, NULL, thread1, &n);
    pthread_join(t0, NULL); pthread_join(t1, NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_SB_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int counts[4] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0;
        pthread_barrier_wait(&bar);
        x = 1; r0 = y;
        pthread_barrier_wait(&bar);
        int idx = r0;
        pthread_barrier_wait(&bar);
        counts[idx * 2 + r1]++;
    }
    printf("0,0:%d\n0,1:%d\n1,0:%d\n1,1:%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    return NULL;
}
void *thread1(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        y = 1; r1 = x;
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 2);
    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread0, &n);
    pthread_create(&t1, NULL, thread1, &n);
    pthread_join(t0, NULL); pthread_join(t1, NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_LB_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0;
        pthread_barrier_wait(&bar);
        r0 = x; y = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread1(void *arg) {
    int counts[4] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r1 = y; x = 1;
        pthread_barrier_wait(&bar);
        counts[r0 * 2 + r1]++;
    }
    printf("0,0:%d\n0,1:%d\n1,0:%d\n1,1:%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 2);
    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread0, &n);
    pthread_create(&t1, NULL, thread1, &n);
    pthread_join(t0, NULL); pthread_join(t1, NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_IRIW_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1, r2, r3;
pthread_barrier_t bar;

void *writer0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0; r0=0; r1=0; r2=0; r3=0;
        pthread_barrier_wait(&bar);
        x = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *writer1(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        y = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *reader0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r0 = x; r1 = y;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *reader1(void *arg) {
    int counts[16] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r2 = y; r3 = x;
        pthread_barrier_wait(&bar);
        counts[r0*8 + r1*4 + r2*2 + r3]++;
    }
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++)
          for (int d = 0; d < 2; d++)
            printf("%d,%d,%d,%d:%d\n", a,b,c,d,
                   counts[a*8+b*4+c*2+d]);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 4);
    pthread_t t[4];
    pthread_create(&t[0], NULL, writer0, &n);
    pthread_create(&t[1], NULL, writer1, &n);
    pthread_create(&t[2], NULL, reader0, &n);
    pthread_create(&t[3], NULL, reader1, &n);
    for (int i = 0; i < 4; i++) pthread_join(t[i], NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_TWO_PLUS_TWO_W_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0;
        pthread_barrier_wait(&bar);
        x = 1; y = 2;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread1(void *arg) {
    int n = *(int*)arg;
    int weak = 0;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        y = 1; x = 2;
        pthread_barrier_wait(&bar);
        if (x == 1 && y == 2) weak++;
    }
    printf("normal:%d\nweak:%d\n", n - weak, weak);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 2);
    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread0, &n);
    pthread_create(&t1, NULL, thread1, &n);
    pthread_join(t0, NULL); pthread_join(t1, NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_RWC_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1, r2;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0; r0=0; r1=0; r2=0;
        pthread_barrier_wait(&bar);
        x = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread1(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r0 = x; y = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread2(void *arg) {
    int counts[8] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r1 = y; r2 = x;
        pthread_barrier_wait(&bar);
        counts[r0*4 + r1*2 + r2]++;
    }
    for (int a = 0; a < 2; a++)
      for (int b = 0; b < 2; b++)
        for (int c = 0; c < 2; c++)
          printf("%d,%d,%d:%d\n", a,b,c, counts[a*4+b*2+c]);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 3);
    pthread_t t[3];
    pthread_create(&t[0], NULL, thread0, &n);
    pthread_create(&t[1], NULL, thread1, &n);
    pthread_create(&t[2], NULL, thread2, &n);
    for (int i = 0; i < 3; i++) pthread_join(t[i], NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_WRC_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int x = 0, y = 0;
int r0, r1;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        x = 0; y = 0; r0=0; r1=0;
        pthread_barrier_wait(&bar);
        x = 1;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread1(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r0 = x; y = r0;
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
void *thread2(void *arg) {
    int counts[4] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        r1 = y; int rx = x;
        pthread_barrier_wait(&bar);
        counts[r1*2 + rx]++;
    }
    printf("0,0:%d\n0,1:%d\n1,0:%d\n1,1:%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 3);
    pthread_t t[3];
    pthread_create(&t[0], NULL, thread0, &n);
    pthread_create(&t[1], NULL, thread1, &n);
    pthread_create(&t[2], NULL, thread2, &n);
    for (int i = 0; i < 3; i++) pthread_join(t[i], NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""

_DEKKER_SOURCE = r"""
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

volatile int flag0 = 0, flag1 = 0;
int r0, r1;
pthread_barrier_t bar;

void *thread0(void *arg) {
    int counts[4] = {0};
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        flag0 = 0; flag1 = 0; r0 = 0; r1 = 0;
        pthread_barrier_wait(&bar);
        flag0 = 1; r0 = flag1;
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
        counts[r0 * 2 + r1]++;
    }
    printf("0,0:%d\n0,1:%d\n1,0:%d\n1,1:%d\n",
           counts[0], counts[1], counts[2], counts[3]);
    return NULL;
}
void *thread1(void *arg) {
    int n = *(int*)arg;
    for (int i = 0; i < n; i++) {
        pthread_barrier_wait(&bar);
        flag1 = 1; r1 = flag0;
        pthread_barrier_wait(&bar);
        pthread_barrier_wait(&bar);
    }
    return NULL;
}
int main(int argc, char **argv) {
    int n = argc > 1 ? atoi(argv[1]) : 100000;
    pthread_barrier_init(&bar, NULL, 2);
    pthread_t t0, t1;
    pthread_create(&t0, NULL, thread0, &n);
    pthread_create(&t1, NULL, thread1, &n);
    pthread_join(t0, NULL); pthread_join(t1, NULL);
    pthread_barrier_destroy(&bar);
    return 0;
}
"""


def _builtin_tests() -> Dict[str, LitmusTestDef]:
    """Return all built-in litmus test definitions with full C source."""
    return {
        "mp": LitmusTestDef(
            name="mp", threads=2, c_source=_MP_SOURCE,
            expected_behaviors={"1,0": False, "0,0": True, "0,1": True, "1,1": True},
            description="Message Passing: can reader see y=1 but x=0?",
        ),
        "sb": LitmusTestDef(
            name="sb", threads=2, c_source=_SB_SOURCE,
            expected_behaviors={"0,0": False, "0,1": True, "1,0": True, "1,1": True},
            description="Store Buffering: can both reads return 0?",
        ),
        "lb": LitmusTestDef(
            name="lb", threads=2, c_source=_LB_SOURCE,
            expected_behaviors={"1,1": False, "0,0": True, "0,1": True, "1,0": True},
            description="Load Buffering: can both reads return 1?",
        ),
        "iriw": LitmusTestDef(
            name="iriw", threads=4, c_source=_IRIW_SOURCE,
            expected_behaviors={"1,0,1,0": False},
            description="Independent Reads of Independent Writes",
        ),
        "2+2w": LitmusTestDef(
            name="2+2w", threads=2, c_source=_TWO_PLUS_TWO_W_SOURCE,
            expected_behaviors={"weak": False, "normal": True},
            description="Two writes per thread, checking store ordering",
        ),
        "rwc": LitmusTestDef(
            name="rwc", threads=3, c_source=_RWC_SOURCE,
            expected_behaviors={"1,1,0": False},
            description="Read-Write Causality",
        ),
        "wrc": LitmusTestDef(
            name="wrc", threads=3, c_source=_WRC_SOURCE,
            expected_behaviors={"1,0": False, "0,0": True, "0,1": True, "1,1": True},
            description="Write-Read Causality",
        ),
        "dekker": LitmusTestDef(
            name="dekker", threads=2, c_source=_DEKKER_SOURCE,
            expected_behaviors={"0,0": False, "0,1": True, "1,0": True, "1,1": True},
            description="Dekker's mutual exclusion idiom",
        ),
    }


def _parse_outcomes(output: str) -> Dict[str, int]:
    """Parse test binary output lines of the form 'label:count'."""
    outcomes: Dict[str, int] = {}
    for line in output.strip().splitlines():
        line = line.strip()
        m = re.match(r"^(.+):(\d+)$", line)
        if m:
            outcomes[m.group(1)] = int(m.group(2))
    return outcomes


def _classify_outcome(outcome: str, test: LitmusTestDef) -> TestStatus:
    """Classify a single outcome key as PASS (expected) or WEAK."""
    if outcome in test.expected_behaviors:
        if test.expected_behaviors[outcome]:
            return TestStatus.PASS
        return TestStatus.WEAK
    # Outcome not listed — treat as unexpected weak behavior
    return TestStatus.WEAK


def _affinity_args(core_list: List[int]) -> str:
    """Return taskset-style CPU affinity argument string."""
    mask = 0
    for c in core_list:
        mask |= 1 << c
    return f"taskset 0x{mask:x}"


class HardwareTester:

    def __init__(self, hosts: Optional[List[str]] = None) -> None:
        self.hosts = hosts if hosts is not None else ["localhost"]
        self.hardware_info = self._detect_hardware()
        self.arch = self.hardware_info.arch
        self._tests = _builtin_tests()

    # ------------------------------------------------------------------
    # Hardware detection
    # ------------------------------------------------------------------

    def _detect_hardware(self) -> HardwareInfo:
        machine = platform.machine().lower()
        arch_map = {
            "x86_64": ArchFamily.X86, "amd64": ArchFamily.X86,
            "i386": ArchFamily.X86, "i686": ArchFamily.X86,
            "aarch64": ArchFamily.AARCH64, "arm64": ArchFamily.AARCH64,
            "armv7l": ArchFamily.ARM, "armv6l": ArchFamily.ARM,
            "riscv64": ArchFamily.RISCV, "ppc64le": ArchFamily.POWER,
            "ppc64": ArchFamily.POWER, "sparc64": ArchFamily.SPARC,
            "mips": ArchFamily.MIPS, "mips64": ArchFamily.MIPS,
        }
        arch = arch_map.get(machine, ArchFamily.X86)
        model_name = "Unknown CPU"
        cores = os.cpu_count() or 1
        threads_per_core = 1
        cache_sizes: Dict[str, int] = {}
        cache_line_size = 64
        numa_nodes = 1
        freq = 0.0

        system = platform.system()
        if system == "Linux":
            model_name, threads_per_core, freq = self._parse_linux_cpu()
            cache_sizes, cache_line_size = self._parse_linux_cache()
            numa_nodes = self._count_linux_numa()
        elif system == "Darwin":
            model_name, freq = self._parse_darwin_cpu()
            cache_sizes, cache_line_size = self._parse_darwin_cache()

        if threads_per_core > 1:
            cores = cores // threads_per_core

        return HardwareInfo(
            arch=arch, model_name=model_name, cores=max(cores, 1),
            threads_per_core=threads_per_core, cache_sizes=cache_sizes,
            cache_line_size=cache_line_size, numa_nodes=numa_nodes,
            frequency_ghz=freq,
        )

    def _parse_linux_cpu(self) -> Tuple[str, int, float]:
        model = "Unknown"
        tpc = 1
        freq = 0.0
        try:
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.startswith("model name"):
                        model = line.split(":", 1)[1].strip()
                    elif line.startswith("siblings"):
                        siblings = int(line.split(":", 1)[1].strip())
                    elif line.startswith("cpu cores"):
                        phys = int(line.split(":", 1)[1].strip())
                        tpc = max(siblings // max(phys, 1), 1) if 'siblings' in dir() else 1
                    elif line.startswith("cpu MHz"):
                        freq = float(line.split(":", 1)[1].strip()) / 1000.0
        except (FileNotFoundError, ValueError):
            pass
        return model, tpc, freq

    def _parse_linux_cache(self) -> Tuple[Dict[str, int], int]:
        sizes: Dict[str, int] = {}
        line_size = 64
        base = "/sys/devices/system/cpu/cpu0/cache"
        try:
            for idx in range(4):
                path = os.path.join(base, f"index{idx}")
                if not os.path.isdir(path):
                    continue
                with open(os.path.join(path, "level")) as f:
                    level = f.read().strip()
                with open(os.path.join(path, "type")) as f:
                    ctype = f.read().strip()
                with open(os.path.join(path, "size")) as f:
                    raw = f.read().strip()
                    val = int(re.sub(r"[^\d]", "", raw)) * 1024
                key = f"L{level}{'d' if ctype == 'Data' else 'i' if ctype == 'Instruction' else ''}"
                sizes[key] = val
                try:
                    with open(os.path.join(path, "coherency_line_size")) as f:
                        line_size = int(f.read().strip())
                except (FileNotFoundError, ValueError):
                    pass
        except (FileNotFoundError, ValueError):
            pass
        return sizes, line_size

    def _count_linux_numa(self) -> int:
        try:
            nodes = [d for d in os.listdir("/sys/devices/system/node")
                     if d.startswith("node")]
            return max(len(nodes), 1)
        except FileNotFoundError:
            return 1

    def _parse_darwin_cpu(self) -> Tuple[str, float]:
        model = "Apple Silicon"
        freq = 0.0
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if out:
                model = out
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.cpufrequency_max"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if out:
                freq = int(out) / 1e9
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        return model, freq

    def _parse_darwin_cache(self) -> Tuple[Dict[str, int], int]:
        sizes: Dict[str, int] = {}
        line_size = 64
        keys = {"hw.l1dcachesize": "L1d", "hw.l1icachesize": "L1i",
                "hw.l2cachesize": "L2", "hw.l3cachesize": "L3"}
        for sysctl_key, label in keys.items():
            try:
                out = subprocess.check_output(
                    ["sysctl", "-n", sysctl_key],
                    stderr=subprocess.DEVNULL, text=True,
                ).strip()
                if out:
                    sizes[label] = int(out)
            except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
                pass
        try:
            out = subprocess.check_output(
                ["sysctl", "-n", "hw.cachelinesize"],
                stderr=subprocess.DEVNULL, text=True,
            ).strip()
            if out:
                line_size = int(out)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError):
            pass
        return sizes, line_size

    # ------------------------------------------------------------------
    # Test generation / compilation / execution
    # ------------------------------------------------------------------

    def _generate_test_program(self, test: LitmusTestDef) -> str:
        return test.c_source

    def _compile_test(self, c_source: str, output_path: str) -> bool:
        src_fd, src_path = tempfile.mkstemp(suffix=".c")
        try:
            with os.fdopen(src_fd, "w") as f:
                f.write(c_source)
            cc = "gcc" if platform.system() != "Darwin" else "cc"
            result = subprocess.run(
                [cc, "-O2", "-pthread", "-o", output_path, src_path],
                capture_output=True, text=True, timeout=30,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
        finally:
            try:
                os.unlink(src_path)
            except OSError:
                pass

    def _run_test_binary(self, binary_path: str, iterations: int) -> Dict[str, int]:
        try:
            result = subprocess.run(
                [binary_path, str(iterations)],
                capture_output=True, text=True, timeout=120,
            )
            if result.returncode != 0:
                return {}
            return _parse_outcomes(result.stdout)
        except subprocess.TimeoutExpired:
            return {"__timeout__": 1}
        except FileNotFoundError:
            return {"__error__": 1}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run_litmus(self, test_name: str,
                   n_iterations: int = 1000000) -> HardwareResult:
        if test_name not in self._tests:
            return HardwareResult(
                test=LitmusTestDef(test_name, 0, "", {}, "unknown"),
                iterations=0, outcomes={}, weak_behaviors_found=[],
                weak_behavior_count=0, total_time_s=0.0,
                status=TestStatus.ERROR,
            )
        test = self._tests[test_name]
        source = self._generate_test_program(test)
        bin_fd, bin_path = tempfile.mkstemp(suffix=".bin")
        os.close(bin_fd)
        try:
            if not self._compile_test(source, bin_path):
                return HardwareResult(
                    test=test, iterations=0, outcomes={},
                    weak_behaviors_found=[], weak_behavior_count=0,
                    total_time_s=0.0, status=TestStatus.ERROR,
                )
            t0 = time.monotonic()
            outcomes = self._run_test_binary(bin_path, n_iterations)
            elapsed = time.monotonic() - t0

            if "__timeout__" in outcomes:
                return HardwareResult(
                    test=test, iterations=n_iterations, outcomes={},
                    weak_behaviors_found=[], weak_behavior_count=0,
                    total_time_s=elapsed, status=TestStatus.TIMEOUT,
                )

            weak_found: List[str] = []
            weak_count = 0
            for key, count in outcomes.items():
                status = _classify_outcome(key, test)
                if status == TestStatus.WEAK and count > 0:
                    weak_found.append(key)
                    weak_count += count

            overall = TestStatus.WEAK if weak_count > 0 else TestStatus.PASS
            return HardwareResult(
                test=test, iterations=n_iterations, outcomes=outcomes,
                weak_behaviors_found=weak_found,
                weak_behavior_count=weak_count,
                total_time_s=elapsed, status=overall,
            )
        finally:
            try:
                os.unlink(bin_path)
            except OSError:
                pass

    def find_weak_behaviors(
        self, test_names: Optional[List[str]] = None,
    ) -> WeakBehaviorReport:
        names = test_names if test_names else list(self._tests.keys())
        results: List[HardwareResult] = []
        for name in names:
            results.append(self.run_litmus(name))
        weak_tests = [r for r in results if r.weak_behavior_count > 0]
        total_weak = sum(r.weak_behavior_count for r in results)
        return WeakBehaviorReport(
            tests=results, total_tests=len(results),
            tests_with_weak=len(weak_tests),
            total_weak_observations=total_weak,
            architecture=str(self.arch),
            hardware_info=self.hardware_info,
        )

    def stress_test(self, test_name: str,
                    duration_s: float = 60.0) -> StressResult:
        test = self._tests.get(test_name)
        if test is None:
            return StressResult(
                test=LitmusTestDef(test_name, 0, "", {}, "unknown"),
                duration_s=0.0, total_iterations=0, outcomes={},
                weak_rate=0.0, peak_weak_rate=0.0, convergence_history=[],
            )
        batch = 100000
        combined: Dict[str, int] = collections.defaultdict(int)
        total_iters = 0
        total_weak = 0
        peak_rate = 0.0
        history: List[Tuple[int, float]] = []

        source = self._generate_test_program(test)
        bin_fd, bin_path = tempfile.mkstemp(suffix=".bin")
        os.close(bin_fd)
        try:
            if not self._compile_test(source, bin_path):
                return StressResult(
                    test=test, duration_s=0.0, total_iterations=0,
                    outcomes={}, weak_rate=0.0, peak_weak_rate=0.0,
                    convergence_history=[],
                )
            start = time.monotonic()
            while time.monotonic() - start < duration_s:
                outcomes = self._run_test_binary(bin_path, batch)
                if "__timeout__" in outcomes or "__error__" in outcomes:
                    break
                total_iters += batch
                batch_weak = 0
                for key, count in outcomes.items():
                    combined[key] += count
                    if _classify_outcome(key, test) == TestStatus.WEAK:
                        batch_weak += count
                total_weak += batch_weak
                batch_rate = batch_weak / batch if batch > 0 else 0.0
                peak_rate = max(peak_rate, batch_rate)
                cumulative_rate = total_weak / total_iters if total_iters else 0.0
                history.append((total_iters, cumulative_rate))

            elapsed = time.monotonic() - start
            avg_rate = total_weak / total_iters if total_iters else 0.0
            return StressResult(
                test=test, duration_s=elapsed, total_iterations=total_iters,
                outcomes=dict(combined), weak_rate=avg_rate,
                peak_weak_rate=peak_rate, convergence_history=history,
            )
        finally:
            try:
                os.unlink(bin_path)
            except OSError:
                pass

    def architecture_fingerprint(self) -> ArchFingerprint:
        diagnostic_tests = ["mp", "sb", "iriw", "lb", "dekker"]
        observed: Dict[str, Set[str]] = {}
        anomalies: List[str] = []

        for name in diagnostic_tests:
            if name not in self._tests:
                continue
            result = self.run_litmus(name, n_iterations=500000)
            observed[name] = set()
            for key, count in result.outcomes.items():
                if count > 0:
                    observed[name].add(key)

        mp_weak = bool(observed.get("mp", set()) & {"1,0"})
        sb_weak = bool(observed.get("sb", set()) & {"0,0"})
        iriw_weak = bool(observed.get("iriw", set()) & {"1,0,1,0"})
        lb_weak = bool(observed.get("lb", set()) & {"1,1"})

        if not mp_weak and not sb_weak and not iriw_weak:
            model, confidence = "SC", 0.7
        elif sb_weak and not mp_weak:
            model, confidence = "TSO", 0.85
        elif mp_weak and not iriw_weak:
            model, confidence = "ARM-like (multi-copy-atomic)", 0.75
        elif mp_weak and iriw_weak:
            model, confidence = "POWER-like (non-multi-copy-atomic)", 0.8
        else:
            model, confidence = "Relaxed", 0.6

        if lb_weak:
            anomalies.append("Load buffering observed — very relaxed model")
        if sb_weak and not mp_weak:
            pass  # normal TSO
        elif mp_weak and sb_weak:
            confidence = min(confidence + 0.05, 1.0)

        return ArchFingerprint(
            hardware_info=self.hardware_info,
            observed_behaviors=observed,
            matches_model=model, confidence=confidence,
            anomalies=anomalies,
        )

    def compare_implementations(
        self, implementations: Dict[str, str],
    ) -> ImplComparison:
        impl_names = list(implementations.keys())
        all_results: Dict[str, Dict[str, HardwareResult]] = {}
        total_weak: Dict[str, int] = collections.defaultdict(int)
        total_time: Dict[str, float] = collections.defaultdict(float)

        test_names = list(self._tests.keys())[:4]

        for impl_name, c_source in implementations.items():
            impl_test = LitmusTestDef(
                name=impl_name, threads=2, c_source=c_source,
                expected_behaviors={}, description=f"Implementation {impl_name}",
            )
            all_results[impl_name] = {}
            bin_fd, bin_path = tempfile.mkstemp(suffix=".bin")
            os.close(bin_fd)
            try:
                compiled = self._compile_test(c_source, bin_path)
                if not compiled:
                    for tn in test_names:
                        all_results[impl_name][tn] = HardwareResult(
                            test=self._tests[tn], iterations=0, outcomes={},
                            weak_behaviors_found=[], weak_behavior_count=0,
                            total_time_s=0.0, status=TestStatus.ERROR,
                        )
                    continue
                for tn in test_names:
                    t0 = time.monotonic()
                    outcomes = self._run_test_binary(bin_path, 100000)
                    elapsed = time.monotonic() - t0
                    test_def = self._tests[tn]
                    weak_found: List[str] = []
                    wc = 0
                    for key, count in outcomes.items():
                        if _classify_outcome(key, test_def) == TestStatus.WEAK and count > 0:
                            weak_found.append(key)
                            wc += count
                    status = TestStatus.WEAK if wc > 0 else TestStatus.PASS
                    result = HardwareResult(
                        test=test_def, iterations=100000, outcomes=outcomes,
                        weak_behaviors_found=weak_found,
                        weak_behavior_count=wc,
                        total_time_s=elapsed, status=status,
                    )
                    all_results[impl_name][tn] = result
                    total_weak[impl_name] += wc
                    total_time[impl_name] += elapsed
            finally:
                try:
                    os.unlink(bin_path)
                except OSError:
                    pass

        fastest = min(impl_names, key=lambda n: total_time.get(n, float("inf")))
        most_correct = min(impl_names, key=lambda n: total_weak.get(n, float("inf")))
        lines = [f"  {n}: weak={total_weak[n]}, time={total_time[n]:.2f}s"
                 for n in impl_names]
        summary = "Implementation comparison:\n" + "\n".join(lines)

        return ImplComparison(
            implementations=impl_names, results=all_results,
            fastest=fastest, most_correct=most_correct, summary=summary,
        )

    def __str__(self) -> str:
        return (f"HardwareTester(hosts={self.hosts}, "
                f"hw={self.hardware_info})")
