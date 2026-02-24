"""
Performance predictor for concurrent programs.

Implements Amdahl's law, Gustafson's law, contention modeling,
cache coherence overhead, false sharing detection, scalability bottleneck
identification, and memory bandwidth modeling.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional, Any
from enum import Enum
import math


class BottleneckType(Enum):
    COMPUTE = "compute"
    MEMORY_BANDWIDTH = "memory_bandwidth"
    LOCK_CONTENTION = "lock_contention"
    CACHE_COHERENCE = "cache_coherence"
    FALSE_SHARING = "false_sharing"
    LOAD_IMBALANCE = "load_imbalance"
    SYNCHRONIZATION = "synchronization"
    IO = "io"


class CacheProtocol(Enum):
    MESI = "MESI"
    MOESI = "MOESI"
    DRAGON = "Dragon"
    MSI = "MSI"


class ArchitectureType(Enum):
    X86 = "x86"
    ARM = "ARM"
    GPU = "GPU"
    POWER = "POWER"
    RISCV = "RISCV"


@dataclass
class AccessPattern:
    variable: str
    thread_id: int
    access_type: str  # "read" or "write"
    frequency: float = 1.0
    cache_line: int = 0


@dataclass
class WorkloadProfile:
    compute_fraction: float = 0.5
    memory_fraction: float = 0.3
    sync_fraction: float = 0.1
    io_fraction: float = 0.1
    parallel_fraction: float = 0.8
    data_size_bytes: int = 1024 * 1024
    ops_per_element: int = 10
    access_patterns: List[AccessPattern] = field(default_factory=list)
    cache_line_size: int = 64
    n_shared_vars: int = 0
    lock_hold_time_ns: float = 100.0
    lock_acquire_time_ns: float = 50.0

    def validate(self):
        total = self.compute_fraction + self.memory_fraction + self.sync_fraction + self.io_fraction
        if abs(total - 1.0) > 0.01:
            scale = 1.0 / total
            self.compute_fraction *= scale
            self.memory_fraction *= scale
            self.sync_fraction *= scale
            self.io_fraction *= scale


@dataclass
class ArchitectureSpec:
    arch_type: ArchitectureType = ArchitectureType.X86
    n_cores: int = 8
    clock_ghz: float = 3.0
    l1_cache_kb: int = 32
    l2_cache_kb: int = 256
    l3_cache_mb: int = 8
    cache_line_bytes: int = 64
    memory_bandwidth_gbps: float = 50.0
    cache_protocol: CacheProtocol = CacheProtocol.MESI
    store_buffer_size: int = 42
    cas_latency_ns: float = 15.0
    fence_latency_ns: float = 30.0
    l1_latency_ns: float = 1.0
    l2_latency_ns: float = 5.0
    l3_latency_ns: float = 20.0
    memory_latency_ns: float = 100.0
    coherence_latency_ns: float = 40.0


@dataclass
class PerfReport:
    predicted_speedup: float
    scalability_limit: int
    bottleneck: BottleneckType
    contention_level: float
    efficiency: float = 0.0
    throughput_ops_per_sec: float = 0.0
    latency_ns: float = 0.0
    cache_miss_rate: float = 0.0
    false_sharing_overhead: float = 0.0
    bandwidth_utilization: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> Dict[str, Any]:
        return {
            "predicted_speedup": round(self.predicted_speedup, 2),
            "scalability_limit": self.scalability_limit,
            "bottleneck": self.bottleneck.value,
            "contention_level": round(self.contention_level, 3),
            "efficiency": round(self.efficiency, 3),
            "cache_miss_rate": round(self.cache_miss_rate, 4),
            "false_sharing_overhead": round(self.false_sharing_overhead, 4),
            "bandwidth_utilization": round(self.bandwidth_utilization, 4),
        }


class AmdahlsLaw:
    """Amdahl's law: speedup from parallel fraction."""

    def speedup(self, parallel_fraction: float, n_threads: int) -> float:
        if parallel_fraction < 0 or parallel_fraction > 1:
            parallel_fraction = max(0, min(1, parallel_fraction))
        serial = 1.0 - parallel_fraction
        return 1.0 / (serial + parallel_fraction / n_threads)

    def max_speedup(self, parallel_fraction: float) -> float:
        serial = 1.0 - parallel_fraction
        if serial <= 0:
            return float('inf')
        return 1.0 / serial

    def efficiency(self, parallel_fraction: float, n_threads: int) -> float:
        return self.speedup(parallel_fraction, n_threads) / n_threads

    def optimal_threads(self, parallel_fraction: float,
                        overhead_per_thread: float = 0.01) -> int:
        best_speedup = 0
        best_n = 1
        for n in range(1, 10000):
            effective_parallel = parallel_fraction - overhead_per_thread * (n - 1)
            if effective_parallel <= 0:
                break
            serial = 1.0 - effective_parallel
            sp = 1.0 / (serial + effective_parallel / n)
            if sp > best_speedup:
                best_speedup = sp
                best_n = n
            elif sp < best_speedup * 0.99:
                break
        return best_n

    def speedup_curve(self, parallel_fraction: float,
                      max_threads: int = 64) -> Dict[str, List[float]]:
        threads = list(range(1, max_threads + 1))
        speedups = [self.speedup(parallel_fraction, n) for n in threads]
        efficiencies = [s / n for s, n in zip(speedups, threads)]
        return {
            "threads": threads,
            "speedup": speedups,
            "efficiency": efficiencies,
            "max_speedup": self.max_speedup(parallel_fraction),
        }


class GustafsonsLaw:
    """Gustafson's law: scaled speedup for data-parallel workloads."""

    def scaled_speedup(self, serial_fraction: float, n_threads: int) -> float:
        return n_threads - serial_fraction * (n_threads - 1)

    def speedup_curve(self, serial_fraction: float,
                      max_threads: int = 64) -> Dict[str, List[float]]:
        threads = list(range(1, max_threads + 1))
        speedups = [self.scaled_speedup(serial_fraction, n) for n in threads]
        return {"threads": threads, "speedup": speedups}


class ContentionModeler:
    """Predict lock contention from access patterns."""

    def predict_contention(self, n_threads: int, lock_hold_time: float,
                           lock_acquire_time: float,
                           arrival_rate: float = 1.0) -> Dict[str, float]:
        # M/M/1 queueing model for lock contention
        service_rate = 1.0 / lock_hold_time
        total_arrival = arrival_rate * n_threads
        utilization = total_arrival / service_rate if service_rate > 0 else 1.0
        utilization = min(utilization, 0.99)

        avg_wait = lock_hold_time * utilization / (1 - utilization) if utilization < 1 else float('inf')
        avg_queue_length = utilization ** 2 / (1 - utilization) if utilization < 1 else float('inf')

        # Probability of finding lock busy
        contention_prob = 1.0 - (1.0 - utilization) ** (n_threads - 1)

        throughput = service_rate * min(1.0, total_arrival / service_rate)

        return {
            "utilization": utilization,
            "avg_wait_time": avg_wait,
            "avg_queue_length": avg_queue_length,
            "contention_probability": contention_prob,
            "throughput": throughput,
            "effective_parallelism": n_threads * (1 - contention_prob),
        }

    def multi_lock_contention(self, n_threads: int, n_locks: int,
                              lock_hold_time: float) -> Dict[str, float]:
        per_lock_threads = n_threads / n_locks
        per_lock_arrival = per_lock_threads / (lock_hold_time * 10)  # rough estimate

        per_lock = self.predict_contention(
            max(2, int(per_lock_threads)), lock_hold_time,
            lock_hold_time * 0.1, per_lock_arrival)

        return {
            "per_lock_contention": per_lock["contention_probability"],
            "total_contention": 1 - (1 - per_lock["contention_probability"]) ** n_locks,
            "effective_parallelism": per_lock["effective_parallelism"] * n_locks / n_threads * n_threads,
            "throughput": per_lock["throughput"] * n_locks,
        }


class CacheCoherenceModeler:
    """Estimate cache coherence overhead from sharing patterns."""

    def __init__(self, arch: ArchitectureSpec = None):
        self.arch = arch or ArchitectureSpec()

    def estimate_overhead(self, access_patterns: List[AccessPattern],
                          n_threads: int) -> Dict[str, float]:
        sharing_matrix = self._build_sharing_matrix(access_patterns, n_threads)
        invalidation_count = self._estimate_invalidations(sharing_matrix, access_patterns)

        coherence_time = invalidation_count * self.arch.coherence_latency_ns
        total_accesses = len(access_patterns)
        miss_rate = invalidation_count / max(total_accesses, 1)

        overhead_fraction = coherence_time / (total_accesses * self.arch.l1_latency_ns + coherence_time)

        return {
            "invalidation_count": invalidation_count,
            "coherence_time_ns": coherence_time,
            "miss_rate": miss_rate,
            "overhead_fraction": overhead_fraction,
            "slowdown_factor": 1.0 + overhead_fraction,
        }

    def _build_sharing_matrix(self, patterns: List[AccessPattern],
                              n_threads: int) -> np.ndarray:
        variables = list(set(p.variable for p in patterns))
        n_vars = len(variables)
        var_idx = {v: i for i, v in enumerate(variables)}

        matrix = np.zeros((n_vars, n_threads))
        for p in patterns:
            vi = var_idx.get(p.variable, 0)
            if p.thread_id < n_threads:
                matrix[vi, p.thread_id] += p.frequency

        return matrix

    def _estimate_invalidations(self, sharing_matrix: np.ndarray,
                                patterns: List[AccessPattern]) -> int:
        n_vars, n_threads = sharing_matrix.shape
        invalidations = 0

        for vi in range(n_vars):
            sharers = np.sum(sharing_matrix[vi] > 0)
            if sharers > 1:
                writes = sum(1 for p in patterns
                             if p.access_type == "write" and
                             p.variable == list(set(p2.variable for p2 in patterns))[min(vi, len(set(p2.variable for p2 in patterns)) - 1)])
                invalidations += int(writes * (sharers - 1))

        return invalidations


class FalseSharingDetector:
    """Detect and predict performance degradation from false sharing."""

    def __init__(self, cache_line_size: int = 64):
        self.cache_line_size = cache_line_size

    def detect(self, access_patterns: List[AccessPattern],
               variable_addresses: Dict[str, int] = None) -> Dict[str, Any]:
        if variable_addresses is None:
            variable_addresses = self._assign_addresses(access_patterns)

        cache_lines = self._map_to_cache_lines(variable_addresses)
        sharing_conflicts = self._find_conflicts(access_patterns, cache_lines)

        total_accesses = len(access_patterns)
        conflict_accesses = sum(c["count"] for c in sharing_conflicts)

        overhead = conflict_accesses / max(total_accesses, 1)

        return {
            "conflicts": sharing_conflicts,
            "n_conflicts": len(sharing_conflicts),
            "overhead_fraction": overhead,
            "affected_cache_lines": len(set(c["cache_line"] for c in sharing_conflicts)),
            "recommendation": self._recommend_fix(sharing_conflicts, variable_addresses),
        }

    def _assign_addresses(self, patterns: List[AccessPattern]) -> Dict[str, int]:
        variables = sorted(set(p.variable for p in patterns))
        addresses = {}
        addr = 0
        for var in variables:
            addresses[var] = addr
            addr += 8  # 8 bytes per variable (worst case: packed)
        return addresses

    def _map_to_cache_lines(self, addresses: Dict[str, int]) -> Dict[str, int]:
        return {var: addr // self.cache_line_size for var, addr in addresses.items()}

    def _find_conflicts(self, patterns: List[AccessPattern],
                        cache_lines: Dict[str, int]) -> List[Dict]:
        conflicts = []
        line_accesses: Dict[int, Dict[int, List[str]]] = {}

        for p in patterns:
            cl = cache_lines.get(p.variable, -1)
            if cl not in line_accesses:
                line_accesses[cl] = {}
            line_accesses[cl].setdefault(p.thread_id, []).append(p.access_type)

        for cl, thread_access in line_accesses.items():
            if len(thread_access) > 1:
                has_write = any("write" in accesses
                                for accesses in thread_access.values())
                if has_write:
                    count = sum(len(a) for a in thread_access.values())
                    conflicts.append({
                        "cache_line": cl,
                        "threads": list(thread_access.keys()),
                        "count": count,
                        "severity": "high" if count > 10 else "medium",
                    })

        return conflicts

    def _recommend_fix(self, conflicts: List[Dict],
                       addresses: Dict[str, int]) -> str:
        if not conflicts:
            return "No false sharing detected"
        if len(conflicts) > 3:
            return "Pad variables to separate cache lines; consider struct-of-arrays layout"
        return "Add padding between frequently accessed variables on same cache line"


class ScalabilityAnalyzer:
    """Identify which resource limits scaling."""

    def analyze(self, workload: WorkloadProfile, arch: ArchitectureSpec,
                thread_counts: List[int] = None) -> Dict[str, Any]:
        if thread_counts is None:
            thread_counts = [1, 2, 4, 8, 16, 32, 64]

        amdahl = AmdahlsLaw()
        contention = ContentionModeler()

        results = {"threads": [], "speedup": [], "bottleneck": [], "efficiency": []}
        bottleneck_counts: Dict[BottleneckType, int] = {}

        for n in thread_counts:
            if n > arch.n_cores:
                # Hyperthreading gives limited benefit
                effective_cores = arch.n_cores + (n - arch.n_cores) * 0.3
            else:
                effective_cores = n

            # Compute-bound limit
            compute_speedup = amdahl.speedup(workload.parallel_fraction, effective_cores)

            # Memory bandwidth limit
            bw_demand = (workload.data_size_bytes * workload.ops_per_element * n /
                         (arch.memory_bandwidth_gbps * 1e9))
            bw_speedup = min(compute_speedup, arch.memory_bandwidth_gbps * 1e9 /
                             max(workload.data_size_bytes * workload.ops_per_element, 1))

            # Lock contention limit
            if workload.sync_fraction > 0:
                lock_result = contention.predict_contention(
                    n, workload.lock_hold_time_ns, workload.lock_acquire_time_ns)
                contention_overhead = lock_result["contention_probability"]
                contention_speedup = compute_speedup * (1 - contention_overhead)
            else:
                contention_speedup = compute_speedup
                contention_overhead = 0

            actual_speedup = min(compute_speedup, bw_speedup, contention_speedup)
            actual_speedup = max(actual_speedup, 1.0)

            # Determine bottleneck
            if bw_speedup < compute_speedup and bw_speedup < contention_speedup:
                bneck = BottleneckType.MEMORY_BANDWIDTH
            elif contention_speedup < compute_speedup:
                bneck = BottleneckType.LOCK_CONTENTION
            else:
                bneck = BottleneckType.COMPUTE

            results["threads"].append(n)
            results["speedup"].append(actual_speedup)
            results["bottleneck"].append(bneck.value)
            results["efficiency"].append(actual_speedup / n)
            bottleneck_counts[bneck] = bottleneck_counts.get(bneck, 0) + 1

        # Find scalability limit (where efficiency drops below 50%)
        scale_limit = thread_counts[-1]
        for i, eff in enumerate(results["efficiency"]):
            if eff < 0.5:
                scale_limit = thread_counts[max(0, i - 1)]
                break

        primary_bottleneck = max(bottleneck_counts.items(), key=lambda x: x[1])[0]
        results["scalability_limit"] = scale_limit
        results["primary_bottleneck"] = primary_bottleneck.value

        return results


class MemoryBandwidthModeler:
    """Predict when bandwidth-bound vs compute-bound."""

    def __init__(self, arch: ArchitectureSpec = None):
        self.arch = arch or ArchitectureSpec()

    def analyze(self, workload: WorkloadProfile,
                n_threads: int) -> Dict[str, float]:
        # Operational intensity (FLOPS per byte)
        ops = workload.data_size_bytes * workload.ops_per_element
        bytes_transferred = workload.data_size_bytes * 2  # read + write
        op_intensity = ops / max(bytes_transferred, 1)

        # Peak GFLOPS
        peak_flops = self.arch.clock_ghz * self.arch.n_cores * 8  # 8 FLOPS/cycle (AVX)

        # Roofline model
        ridge_point = peak_flops / self.arch.memory_bandwidth_gbps
        achieved_flops = min(
            peak_flops * min(n_threads / self.arch.n_cores, 1.0),
            op_intensity * self.arch.memory_bandwidth_gbps
        )

        is_bandwidth_bound = op_intensity < ridge_point

        bandwidth_usage = (bytes_transferred * n_threads /
                           (self.arch.memory_bandwidth_gbps * 1e9))

        return {
            "operational_intensity": op_intensity,
            "ridge_point": ridge_point,
            "peak_gflops": peak_flops,
            "achieved_gflops": achieved_flops,
            "is_bandwidth_bound": is_bandwidth_bound,
            "bandwidth_utilization": min(1.0, bandwidth_usage),
            "bottleneck": "memory_bandwidth" if is_bandwidth_bound else "compute",
        }

    def roofline_curve(self, n_threads: int,
                       intensity_range: Optional[List[float]] = None) -> Dict[str, List[float]]:
        if intensity_range is None:
            intensity_range = np.logspace(-2, 3, 50).tolist()

        peak_flops = self.arch.clock_ghz * self.arch.n_cores * 8
        effective_peak = peak_flops * min(n_threads / self.arch.n_cores, 1.0)

        achieved = []
        for oi in intensity_range:
            perf = min(effective_peak, oi * self.arch.memory_bandwidth_gbps)
            achieved.append(perf)

        return {
            "operational_intensity": intensity_range,
            "achieved_gflops": achieved,
            "peak_gflops": effective_peak,
            "memory_bandwidth_gbps": self.arch.memory_bandwidth_gbps,
        }


class ConcurrencyPerformancePredictor:
    """Main predictor: orchestrates all performance analysis."""

    def __init__(self, arch: ArchitectureSpec = None):
        self.arch = arch or ArchitectureSpec()
        self.amdahl = AmdahlsLaw()
        self.gustafson = GustafsonsLaw()
        self.contention = ContentionModeler()
        self.cache_coherence = CacheCoherenceModeler(self.arch)
        self.false_sharing = FalseSharingDetector(self.arch.cache_line_bytes)
        self.scalability = ScalabilityAnalyzer()
        self.bandwidth = MemoryBandwidthModeler(self.arch)

    def predict(self, workload: WorkloadProfile, n_threads: int,
                architecture: ArchitectureSpec = None) -> PerfReport:
        arch = architecture or self.arch
        workload.validate()

        # Amdahl's law baseline
        base_speedup = self.amdahl.speedup(workload.parallel_fraction, n_threads)

        # Contention modeling
        lock_result = self.contention.predict_contention(
            n_threads, workload.lock_hold_time_ns, workload.lock_acquire_time_ns)
        contention_level = lock_result["contention_probability"]

        # Cache coherence
        cache_result = self.cache_coherence.estimate_overhead(
            workload.access_patterns, n_threads)
        cache_overhead = cache_result.get("overhead_fraction", 0)

        # False sharing
        fs_result = self.false_sharing.detect(workload.access_patterns)
        fs_overhead = fs_result.get("overhead_fraction", 0)

        # Memory bandwidth
        bw_result = self.bandwidth.analyze(workload, n_threads)
        bw_util = bw_result.get("bandwidth_utilization", 0)

        # Combined speedup with overheads
        overhead_factor = (1 - contention_level) * (1 - cache_overhead) * (1 - fs_overhead)
        predicted_speedup = base_speedup * max(overhead_factor, 0.1)

        # Determine bottleneck
        overheads = {
            BottleneckType.LOCK_CONTENTION: contention_level,
            BottleneckType.CACHE_COHERENCE: cache_overhead,
            BottleneckType.FALSE_SHARING: fs_overhead,
            BottleneckType.MEMORY_BANDWIDTH: bw_util,
        }
        bottleneck = max(overheads.items(), key=lambda x: x[1])[0]
        if max(overheads.values()) < 0.1:
            bottleneck = BottleneckType.COMPUTE

        # Scalability limit
        scale_result = self.scalability.analyze(workload, arch)
        scale_limit = scale_result.get("scalability_limit", n_threads)

        efficiency = predicted_speedup / n_threads

        return PerfReport(
            predicted_speedup=predicted_speedup,
            scalability_limit=scale_limit,
            bottleneck=bottleneck,
            contention_level=contention_level,
            efficiency=efficiency,
            cache_miss_rate=cache_result.get("miss_rate", 0),
            false_sharing_overhead=fs_overhead,
            bandwidth_utilization=bw_util,
            details={
                "amdahl_speedup": base_speedup,
                "contention": lock_result,
                "cache_coherence": cache_result,
                "false_sharing": fs_result,
                "bandwidth": bw_result,
                "scalability": scale_result,
            },
        )

    def predict_speedup_curve(self, workload: WorkloadProfile,
                              max_threads: int = 64) -> Dict[str, List]:
        threads = [2 ** i for i in range(int(math.log2(max_threads)) + 1)]
        threads = [t for t in threads if t <= max_threads]
        if 1 not in threads:
            threads = [1] + threads

        speedups = []
        bottlenecks = []
        efficiencies = []

        for n in threads:
            report = self.predict(workload, n)
            speedups.append(report.predicted_speedup)
            bottlenecks.append(report.bottleneck.value)
            efficiencies.append(report.efficiency)

        return {
            "threads": threads,
            "speedup": speedups,
            "efficiency": efficiencies,
            "bottleneck": bottlenecks,
        }

    def compare_architectures(self, workload: WorkloadProfile,
                              n_threads: int,
                              architectures: Dict[str, ArchitectureSpec]
                              ) -> Dict[str, PerfReport]:
        results = {}
        for name, arch in architectures.items():
            results[name] = self.predict(workload, n_threads, arch)
        return results


class QueueingModel:
    """M/M/c queueing model for lock contention analysis."""

    def __init__(self, n_servers: int = 1):
        self.n_servers = n_servers

    def utilization(self, arrival_rate: float, service_rate: float) -> float:
        rho = arrival_rate / (self.n_servers * service_rate)
        return min(rho, 0.999)

    def avg_queue_length(self, arrival_rate: float, service_rate: float) -> float:
        rho = self.utilization(arrival_rate, service_rate)
        if self.n_servers == 1:
            return rho ** 2 / (1 - rho)
        # Erlang-C approximation for M/M/c
        c = self.n_servers
        a = arrival_rate / service_rate
        p0 = self._compute_p0(a, c)
        pc = (a ** c / math.factorial(c)) * p0
        lq = pc * rho / ((1 - rho) ** 2) if rho < 1 else float('inf')
        return lq

    def avg_wait_time(self, arrival_rate: float, service_rate: float) -> float:
        lq = self.avg_queue_length(arrival_rate, service_rate)
        return lq / arrival_rate if arrival_rate > 0 else 0

    def avg_response_time(self, arrival_rate: float, service_rate: float) -> float:
        wq = self.avg_wait_time(arrival_rate, service_rate)
        return wq + 1.0 / service_rate

    def throughput(self, arrival_rate: float, service_rate: float) -> float:
        return min(arrival_rate, self.n_servers * service_rate)

    def _compute_p0(self, a: float, c: int) -> float:
        """Compute probability of empty system for M/M/c."""
        s = 0.0
        for n in range(c):
            s += (a ** n) / math.factorial(n)
        rho = a / c
        if rho < 1:
            s += (a ** c) / (math.factorial(c) * (1 - rho))
        else:
            s += (a ** c) / math.factorial(c)
        return 1.0 / s if s > 0 else 0


class CacheModel:
    """Model cache behavior for concurrent programs."""

    def __init__(self, arch: ArchitectureSpec = None):
        self.arch = arch or ArchitectureSpec()

    def estimate_working_set(self, data_size_bytes: int, n_threads: int,
                             sharing_degree: float = 0.5) -> Dict[str, float]:
        """Estimate per-thread working set and cache pressure."""
        private_data = data_size_bytes * (1 - sharing_degree) / n_threads
        shared_data = data_size_bytes * sharing_degree

        per_thread_ws = private_data + shared_data
        l1_fit = per_thread_ws <= self.arch.l1_cache_kb * 1024
        l2_fit = per_thread_ws <= self.arch.l2_cache_kb * 1024
        l3_fit = per_thread_ws * n_threads <= self.arch.l3_cache_mb * 1024 * 1024

        if l1_fit:
            avg_latency = self.arch.l1_latency_ns
        elif l2_fit:
            avg_latency = self.arch.l2_latency_ns
        elif l3_fit:
            avg_latency = self.arch.l3_latency_ns
        else:
            avg_latency = self.arch.memory_latency_ns

        return {
            "per_thread_working_set": per_thread_ws,
            "fits_l1": l1_fit,
            "fits_l2": l2_fit,
            "fits_l3": l3_fit,
            "estimated_avg_latency_ns": avg_latency,
            "shared_data_fraction": sharing_degree,
        }

    def estimate_miss_rate(self, data_size_bytes: int, cache_size_bytes: int,
                           associativity: int = 8) -> float:
        """Estimate cache miss rate using simplified model."""
        if data_size_bytes <= cache_size_bytes:
            return 0.01  # compulsory misses only
        ratio = cache_size_bytes / data_size_bytes
        # Power-law miss rate model
        miss_rate = (1 - ratio) ** 0.5
        return min(miss_rate, 1.0)

    def coherence_traffic(self, n_threads: int, write_sharing_fraction: float,
                          access_rate: float) -> Dict[str, float]:
        """Estimate coherence traffic between caches."""
        invalidations_per_sec = access_rate * write_sharing_fraction * (n_threads - 1)
        bytes_per_invalidation = self.arch.cache_line_bytes
        coherence_bandwidth = invalidations_per_sec * bytes_per_invalidation

        return {
            "invalidations_per_sec": invalidations_per_sec,
            "coherence_bandwidth_bytes": coherence_bandwidth,
            "fraction_of_memory_bw": coherence_bandwidth / (self.arch.memory_bandwidth_gbps * 1e9),
        }


class NUMAModel:
    """Model NUMA effects on concurrent program performance."""

    def __init__(self, n_nodes: int = 2, local_latency_ns: float = 100,
                 remote_latency_ns: float = 300):
        self.n_nodes = n_nodes
        self.local_latency = local_latency_ns
        self.remote_latency = remote_latency_ns

    def predict_latency(self, n_threads: int, data_locality: float = 0.8) -> float:
        """Predict average memory access latency considering NUMA placement."""
        threads_per_node = n_threads / self.n_nodes
        local_fraction = data_locality
        remote_fraction = 1.0 - local_fraction

        avg_latency = (local_fraction * self.local_latency +
                       remote_fraction * self.remote_latency)
        return avg_latency

    def numa_speedup_factor(self, n_threads: int, data_locality: float = 0.8) -> float:
        """Compute speedup factor due to NUMA effects."""
        avg_latency = self.predict_latency(n_threads, data_locality)
        ideal_latency = self.local_latency
        return ideal_latency / avg_latency

    def optimal_thread_placement(self, n_threads: int,
                                 data_distribution: Dict[int, float]) -> Dict[int, int]:
        """Suggest optimal thread-to-node placement."""
        placement = {}
        threads_remaining = n_threads
        for node in range(self.n_nodes):
            fraction = data_distribution.get(node, 1.0 / self.n_nodes)
            node_threads = max(1, int(round(n_threads * fraction)))
            node_threads = min(node_threads, threads_remaining)
            for t in range(node_threads):
                tid = n_threads - threads_remaining + t
                placement[tid] = node
            threads_remaining -= node_threads
        return placement


class SynchronizationCostModel:
    """Model the cost of different synchronization primitives."""

    def __init__(self, arch: ArchitectureSpec = None):
        self.arch = arch or ArchitectureSpec()

    def mutex_cost(self, n_threads: int, contention: float = 0.5) -> Dict[str, float]:
        """Estimate cost of mutex operations under contention."""
        base_lock_ns = self.arch.cas_latency_ns * 2
        contention_wait = base_lock_ns * contention * (n_threads - 1)
        total_cost = base_lock_ns + contention_wait

        return {
            "lock_acquire_ns": total_cost,
            "lock_release_ns": base_lock_ns * 0.5,
            "total_lock_unlock_ns": total_cost + base_lock_ns * 0.5,
            "contention_overhead": contention_wait,
        }

    def cas_cost(self, n_threads: int, success_probability: float = 0.8) -> Dict[str, float]:
        """Estimate cost of CAS operations."""
        base_cas_ns = self.arch.cas_latency_ns
        avg_attempts = 1.0 / success_probability if success_probability > 0 else float('inf')
        total_cost = base_cas_ns * avg_attempts

        return {
            "single_cas_ns": base_cas_ns,
            "avg_attempts": avg_attempts,
            "avg_total_ns": total_cost,
            "success_probability": success_probability,
        }

    def barrier_cost(self, n_threads: int) -> Dict[str, float]:
        """Estimate cost of barrier synchronization."""
        # Tree-based barrier: O(log n) rounds
        rounds = math.ceil(math.log2(max(n_threads, 2)))
        per_round_ns = self.arch.cas_latency_ns + self.arch.coherence_latency_ns
        total_cost = rounds * per_round_ns

        return {
            "barrier_rounds": rounds,
            "per_round_ns": per_round_ns,
            "total_barrier_ns": total_cost,
            "load_imbalance_wait_ns": total_cost * 0.5,  # average idle time
        }

    def fence_cost(self, fence_type: str = "full") -> float:
        """Estimate cost of memory fence."""
        costs = {
            "full": self.arch.fence_latency_ns,
            "store_store": self.arch.fence_latency_ns * 0.3,
            "load_load": self.arch.fence_latency_ns * 0.2,
            "store_load": self.arch.fence_latency_ns * 0.8,
            "acquire": self.arch.fence_latency_ns * 0.5,
            "release": self.arch.fence_latency_ns * 0.5,
        }
        return costs.get(fence_type, self.arch.fence_latency_ns)
