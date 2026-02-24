"""Hardware recommendations for neural network training.

Provides GPU selection, memory budgeting, multi-GPU scaling analysis,
and cloud cost comparison for finite-width neural network experiments.

Example
-------
>>> model = {"input_dim": 784, "width": 512, "depth": 4, "init_scale": 1.0}
>>> dataset = {"n_samples": 60000, "input_dim": 784, "n_classes": 10, "epochs": 100}
>>> rec = recommend_hardware(model, dataset, budget=50.0)
>>> rec.gpu_type in GPU_SPECS
True
>>> mem = gpu_memory_check(model, batch_size=64)
>>> mem.total_memory_mb > 0
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray


# ======================================================================
# Constants
# ======================================================================

GPU_SPECS: Dict[str, Dict[str, float]] = {
    "T4": {"memory_gb": 16, "tflops_fp32": 8.1, "tflops_fp16": 65,
            "bandwidth_gb_s": 320, "cost_per_hour": 0.35},
    "V100": {"memory_gb": 32, "tflops_fp32": 15.7, "tflops_fp16": 125,
             "bandwidth_gb_s": 900, "cost_per_hour": 3.06},
    "A100_40": {"memory_gb": 40, "tflops_fp32": 19.5, "tflops_fp16": 312,
                "bandwidth_gb_s": 1555, "cost_per_hour": 4.10},
    "A100_80": {"memory_gb": 80, "tflops_fp32": 19.5, "tflops_fp16": 312,
                "bandwidth_gb_s": 2039, "cost_per_hour": 5.50},
    "H100": {"memory_gb": 80, "tflops_fp32": 51, "tflops_fp16": 990,
             "bandwidth_gb_s": 3350, "cost_per_hour": 8.50},
    "RTX_4090": {"memory_gb": 24, "tflops_fp32": 82.6, "tflops_fp16": 165,
                 "bandwidth_gb_s": 1008, "cost_per_hour": 0.74},
}

INTERCONNECT_BANDWIDTH: Dict[str, float] = {
    "pcie": 32.0, "nvlink": 600.0, "infiniband": 200.0,
}

PROVIDER_MULTIPLIERS: Dict[str, float] = {
    "aws": 1.0, "gcp": 0.95, "azure": 0.98, "lambda": 0.70,
}


# ======================================================================
# Data structures
# ======================================================================


@dataclass
class HardwareRecommendation:
    """Hardware recommendation for a training job."""

    gpu_type: str
    n_gpus: int
    batch_size: int
    estimated_time_hours: float
    estimated_cost_usd: float
    explanation: str
    alternatives: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class MemoryCheck:
    """Result of a GPU memory feasibility check."""

    fits: bool
    model_memory_mb: float
    activation_memory_mb: float
    optimizer_memory_mb: float
    total_memory_mb: float
    gpu_memory_mb: float
    utilization: float
    max_batch_size: int
    recommendation: str


@dataclass
class ScalingPrediction:
    """Multi-GPU scaling prediction."""

    n_gpus: int
    speedup: float
    efficiency: float
    communication_overhead: float
    recommended: bool
    explanation: str


@dataclass
class CostTable:
    """Cloud cost comparison table."""

    entries: List[Dict[str, Any]]
    cheapest_idx: int
    fastest_idx: int
    best_value_idx: int
    summary: str


# ======================================================================
# Helper utilities
# ======================================================================


def _compute_n_params(model: Dict[str, Any]) -> int:
    """Return the total number of trainable parameters."""
    w, d = model["width"], model["depth"]
    nc = model.get("n_classes", 10)
    return model["input_dim"] * w + (d - 1) * w * w + w * nc


def _compute_total_flops(model: Dict[str, Any], dataset: Dict[str, Any]) -> float:
    """Return total training FLOPs (forward + backward + optimizer)."""
    n_params = _compute_n_params({**model, "n_classes": dataset.get("n_classes", 10)})
    return 6.0 * n_params * dataset["n_samples"] * dataset["epochs"]


def _memory_breakdown_mb(
    model: Dict[str, Any], batch_size: int, mixed_precision: bool = False,
) -> tuple[float, float, float]:
    """Return ``(model_mb, activation_mb, optimizer_mb)``."""
    n_params = _compute_n_params(model)
    bpp = 2.0 if mixed_precision else 4.0
    model_mb = n_params * bpp / (1024 * 1024)
    act_mb = batch_size * model["width"] * model["depth"] * 4.0 / (1024 * 1024)
    opt_mb = n_params * 8.0 / (1024 * 1024)  # Adam: m + v
    return model_mb, act_mb, opt_mb


def _utilization(n_params: int) -> float:
    return 0.3 if n_params < 1e7 else 0.5


# ======================================================================
# Public API
# ======================================================================


def recommend_hardware(
    model: Dict[str, Any], dataset: Dict[str, Any], budget: float = 100.0,
) -> HardwareRecommendation:
    """Recommend hardware for a training job.

    Parameters
    ----------
    model : Dict
        Keys: ``input_dim``, ``width``, ``depth``, ``init_scale``.
    dataset : Dict
        Keys: ``n_samples``, ``input_dim``, ``n_classes``, ``epochs``.
    budget : float
        Maximum budget in USD.

    Returns
    -------
    HardwareRecommendation
    """
    enriched = {**model, "n_classes": dataset.get("n_classes", 10)}
    total_flops = _compute_total_flops(model, dataset)
    n_params = _compute_n_params(enriched)
    sorted_gpus = sorted(GPU_SPECS.items(), key=lambda x: x[1]["cost_per_hour"])

    best: Optional[Dict[str, Any]] = None
    alternatives: List[Dict[str, Any]] = []

    for gpu_name, specs in sorted_gpus:
        mem_mb, act_mb, opt_mb = _memory_breakdown_mb(enriched, 64)
        if mem_mb + act_mb + opt_mb > specs["memory_gb"] * 1024:
            continue
        eff_flops = specs["tflops_fp32"] * _utilization(n_params) * 1e12
        time_h = total_flops / eff_flops / 3600.0
        cost = time_h * specs["cost_per_hour"]
        n_gpus = 1
        if cost > budget:
            n_gpus = max(1, math.ceil(cost / budget))
            time_h /= n_gpus * 0.85
            cost = time_h * specs["cost_per_hour"] * n_gpus

        entry = {"gpu_type": gpu_name, "n_gpus": n_gpus,
                 "time_hours": round(time_h, 2), "cost_usd": round(cost, 2)}
        if best is None and cost <= budget:
            best = entry
        else:
            alternatives.append(entry)

    if best is None:
        best = alternatives.pop(0) if alternatives else {
            "gpu_type": "H100", "n_gpus": 1, "time_hours": 1.0, "cost_usd": 8.50}
    explanation = (
        f"Recommended {best['gpu_type']} x{best['n_gpus']} for "
        f"~{best['time_hours']:.1f}h at ${best['cost_usd']:.2f}."
    )
    return HardwareRecommendation(
        gpu_type=best["gpu_type"], n_gpus=best["n_gpus"], batch_size=64,
        estimated_time_hours=best["time_hours"],
        estimated_cost_usd=best["cost_usd"],
        explanation=explanation, alternatives=alternatives[:2],
    )


def gpu_memory_check(
    model: Dict[str, Any], batch_size: int,
    gpu_type: str = "A100_40", mixed_precision: bool = False,
) -> MemoryCheck:
    """Check whether a model configuration fits in GPU memory.

    Parameters
    ----------
    model : Dict
        Keys: ``input_dim``, ``width``, ``depth``.
    batch_size : int
        Training batch size.
    gpu_type : str
        GPU type key in ``GPU_SPECS``.
    mixed_precision : bool
        Whether mixed-precision training is used.

    Returns
    -------
    MemoryCheck
    """
    specs = GPU_SPECS[gpu_type]
    gpu_mem_mb = specs["memory_gb"] * 1024.0
    model_mb, act_mb, opt_mb = _memory_breakdown_mb(model, batch_size, mixed_precision)
    total_mb = model_mb + act_mb + opt_mb
    util = total_mb / gpu_mem_mb
    fits = total_mb <= gpu_mem_mb

    lo, hi = 1, batch_size * 64
    while lo < hi:
        mid = (lo + hi + 1) // 2
        _, a, _ = _memory_breakdown_mb(model, mid, mixed_precision)
        if model_mb + a + opt_mb <= gpu_mem_mb:
            lo = mid
        else:
            hi = mid - 1

    if fits and util < 0.7:
        rec = f"Good fit on {gpu_type}. Consider increasing batch size to {lo}."
    elif fits:
        rec = f"Tight fit on {gpu_type} ({util:.0%} utilization)."
    else:
        rec = f"Does not fit on {gpu_type}. Reduce batch size to <={lo} or use a larger GPU."

    return MemoryCheck(
        fits=fits, model_memory_mb=round(model_mb, 2),
        activation_memory_mb=round(act_mb, 2), optimizer_memory_mb=round(opt_mb, 2),
        total_memory_mb=round(total_mb, 2), gpu_memory_mb=round(gpu_mem_mb, 2),
        utilization=round(util, 4), max_batch_size=lo, recommendation=rec,
    )


def multi_gpu_benefit(
    model: Dict[str, Any], n_gpus: int, interconnect: str = "pcie",
) -> ScalingPrediction:
    """Predict speedup from multi-GPU data-parallel training.

    Parameters
    ----------
    model : Dict
        Keys: ``input_dim``, ``width``, ``depth``.
    n_gpus : int
        Number of GPUs to evaluate.
    interconnect : str
        One of ``"pcie"``, ``"nvlink"``, ``"infiniband"``.

    Returns
    -------
    ScalingPrediction
    """
    n_params = _compute_n_params(model)
    model_gb = n_params * 4.0 / (1024 ** 3)
    bw = INTERCONNECT_BANDWIDTH[interconnect]

    # Allreduce: 2*(n-1)/n * model_size / bandwidth
    comm = 2.0 * (n_gpus - 1) / n_gpus * model_gb / bw
    single = 1.0
    multi_total = single / n_gpus + comm

    speedup = single / multi_total
    efficiency = speedup / n_gpus
    overhead = comm / multi_total if multi_total > 0 else 0.0
    recommended = efficiency > 0.7

    tag = "Recommended." if recommended else "Communication overhead too high."
    explanation = (
        f"{n_gpus} GPUs via {interconnect}: {speedup:.2f}x speedup "
        f"({efficiency:.0%} efficiency). {tag}"
    )
    return ScalingPrediction(
        n_gpus=n_gpus, speedup=round(speedup, 3), efficiency=round(efficiency, 4),
        communication_overhead=round(overhead, 4), recommended=recommended,
        explanation=explanation,
    )


def cloud_cost_comparison(
    model: Dict[str, Any], dataset: Dict[str, Any],
    providers: Optional[List[str]] = None,
) -> CostTable:
    """Compare training costs across cloud providers and GPU types.

    Parameters
    ----------
    model : Dict
        Model specification.
    dataset : Dict
        Dataset specification.
    providers : List[str], optional
        Defaults to ``["aws", "gcp", "azure", "lambda"]``.

    Returns
    -------
    CostTable
    """
    if providers is None:
        providers = ["aws", "gcp", "azure", "lambda"]

    total_flops = _compute_total_flops(model, dataset)
    enriched = {**model, "n_classes": dataset.get("n_classes", 10)}
    n_params = _compute_n_params(enriched)
    entries: List[Dict[str, Any]] = []

    for provider in providers:
        mult = PROVIDER_MULTIPLIERS.get(provider, 1.0)
        for gpu_name, specs in GPU_SPECS.items():
            m, a, o = _memory_breakdown_mb(enriched, 64)
            if m + a + o > specs["memory_gb"] * 1024:
                continue
            eff = specs["tflops_fp32"] * _utilization(n_params) * 1e12
            time_h = total_flops / eff / 3600.0
            cost = time_h * specs["cost_per_hour"] * mult
            ppc = 1.0 / cost if cost > 0 else float("inf")
            entries.append({"provider": provider, "gpu_type": gpu_name,
                            "time_hours": round(time_h, 2),
                            "cost_usd": round(cost, 2),
                            "perf_per_cost": round(ppc, 4)})

    if not entries:
        return CostTable([], -1, -1, -1, "No feasible configurations found.")

    cheapest = min(range(len(entries)), key=lambda i: entries[i]["cost_usd"])
    fastest = min(range(len(entries)), key=lambda i: entries[i]["time_hours"])
    best_val = max(range(len(entries)), key=lambda i: entries[i]["perf_per_cost"])
    c, f = entries[cheapest], entries[fastest]
    summary = (
        f"Cheapest: {c['provider']}/{c['gpu_type']} at ${c['cost_usd']:.2f}. "
        f"Fastest: {f['provider']}/{f['gpu_type']} in {f['time_hours']:.1f}h."
    )
    return CostTable(entries=entries, cheapest_idx=cheapest,
                     fastest_idx=fastest, best_value_idx=best_val, summary=summary)


def training_time_estimate(
    model: Dict[str, Any], dataset: Dict[str, Any], hardware: str = "A100_40",
) -> float:
    """Estimate training time in hours.

    Parameters
    ----------
    model : Dict
        Keys: ``input_dim``, ``width``, ``depth``.
    dataset : Dict
        Keys: ``n_samples``, ``epochs``.
    hardware : str
        GPU type key in ``GPU_SPECS``.

    Returns
    -------
    float
        Estimated training time in hours.
    """
    specs = GPU_SPECS[hardware]
    total_flops = _compute_total_flops(model, dataset)
    enriched = {**model, "n_classes": dataset.get("n_classes", 10)}
    util = _utilization(_compute_n_params(enriched))
    return total_flops / (specs["tflops_fp32"] * util * 1e12) / 3600.0
