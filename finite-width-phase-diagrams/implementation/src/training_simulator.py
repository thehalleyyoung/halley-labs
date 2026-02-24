"""Simulate training dynamics without GPUs using NTK theory predictions.

Provides lightweight training simulation based on Neural Tangent Kernel
eigenvalue decomposition, enabling hyperparameter search and cost
estimation without requiring actual GPU hardware.  The simulator models
loss decay through spectral modes of the NTK Gram matrix, capturing the
qualitative difference between lazy (kernel) and rich (feature-learning)
regimes.

Example
-------
>>> from phase_diagrams.training_simulator import simulate_training
>>> model = {"input_dim": 784, "width": 512, "depth": 3, "init_scale": 1.0}
>>> data = {"n_samples": 1000, "input_dim": 784, "n_classes": 10, "noise_level": 0.1}
>>> cfg = {"lr": 0.01, "epochs": 100, "batch_size": 32, "optimizer": "sgd"}
>>> run = simulate_training(model, data, cfg)
>>> print(run.regime, run.converged)
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import NDArray

from .api import Regime


# ======================================================================
# Data classes
# ======================================================================

@dataclass
class SimulatedRun:
    """Result of a single simulated training run.

    Attributes
    ----------
    epochs : int
        Number of training epochs executed.
    final_loss : float
        Loss value at the final epoch.
    final_accuracy : float
        Accuracy at the final epoch.
    loss_curve : NDArray
        Per-epoch loss values, shape ``(epochs,)``.
    accuracy_curve : NDArray
        Per-epoch accuracy values, shape ``(epochs,)``.
    regime : str
        Detected training regime (``"lazy"`` or ``"rich"``).
    converged : bool
        Whether the run converged (loss < 1% of initial loss).
    convergence_epoch : int
        Epoch at which convergence was first reached, or ``-1``.
    wall_time_estimate : float
        Estimated wall-clock time in seconds on reference hardware.
    """

    epochs: int = 0
    final_loss: float = float("inf")
    final_accuracy: float = 0.0
    loss_curve: NDArray = field(default_factory=lambda: np.array([]))
    accuracy_curve: NDArray = field(default_factory=lambda: np.array([]))
    regime: str = "lazy"
    converged: bool = False
    convergence_epoch: int = -1
    wall_time_estimate: float = 0.0


@dataclass
class GridResult:
    """Result of a grid search over hyperparameters.

    Attributes
    ----------
    configs : List[Dict]
        All evaluated configurations.
    results : List[SimulatedRun]
        Corresponding simulated runs.
    best_config : Dict
        Configuration that achieved the lowest final loss.
    best_result : SimulatedRun
        SimulatedRun for the best configuration.
    param_importances : Dict[str, float]
        Variance-decomposition importance per parameter.
    """

    configs: List[Dict] = field(default_factory=list)
    results: List[SimulatedRun] = field(default_factory=list)
    best_config: Dict = field(default_factory=dict)
    best_result: SimulatedRun = field(default_factory=SimulatedRun)
    param_importances: Dict[str, float] = field(default_factory=dict)


@dataclass
class HyperbandResult:
    """Result of Hyperband-style hyperparameter search.

    Attributes
    ----------
    best_config : Dict
        Best configuration found.
    best_result : SimulatedRun
        SimulatedRun for the best configuration.
    configs_evaluated : int
        Total number of configurations evaluated.
    total_budget_used : float
        Total epoch-budget consumed across all brackets.
    bracket_results : List[Dict]
        Per-bracket summary dictionaries.
    """

    best_config: Dict = field(default_factory=dict)
    best_result: SimulatedRun = field(default_factory=SimulatedRun)
    configs_evaluated: int = 0
    total_budget_used: float = 0.0
    bracket_results: List[Dict] = field(default_factory=list)


@dataclass
class CostEstimate:
    """Cloud cost estimate for a training job.

    Attributes
    ----------
    provider : str
        Cloud provider name.
    gpu_type : str
        GPU model identifier.
    hours : float
        Estimated training time in hours.
    cost_usd : float
        On-demand cost in US dollars.
    spot_cost_usd : float
        Spot / preemptible cost in US dollars.
    breakdown : Dict[str, float]
        Cost breakdown by category (compute, storage, network).
    """

    provider: str = ""
    gpu_type: str = ""
    hours: float = 0.0
    cost_usd: float = 0.0
    spot_cost_usd: float = 0.0
    breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class TimeEstimate:
    """Wall-clock time estimate for a training job.

    Attributes
    ----------
    hardware : str
        Hardware identifier used for estimation.
    hours : float
        Estimated training time in hours.
    throughput_samples_per_sec : float
        Expected throughput in samples per second.
    bottleneck : str
        Primary bottleneck (``"compute"`` or ``"memory"``).
    gpu_utilization : float
        Expected GPU utilization fraction (0-1).
    """

    hardware: str = ""
    hours: float = 0.0
    throughput_samples_per_sec: float = 0.0
    bottleneck: str = "compute"
    gpu_utilization: float = 0.0


# ======================================================================
# Hardware reference tables
# ======================================================================

_GPU_TFLOPS: Dict[str, float] = {
    "V100": 15.7,
    "A100": 312.0,
    "H100": 990.0,
    "T4": 8.1,
    "CPU": 0.1,
}

_GPU_PRICE_PER_HOUR: Dict[str, float] = {
    "V100": 3.06,
    "A100": 4.10,
    "H100": 8.50,
    "T4": 0.35,
}

_GPU_MEMORY_GB: Dict[str, float] = {
    "V100": 16.0,
    "A100": 80.0,
    "H100": 80.0,
    "T4": 16.0,
    "CPU": 64.0,
}

_SPOT_DISCOUNT: float = 0.30


# ======================================================================
# Internal helpers
# ======================================================================

def _compute_flops_per_sample(model: Dict) -> float:
    """Estimate FLOPs for a single forward + backward pass.

    Uses *2 * width^2 * depth* as a rough linear-layer FLOP count
    and triples for forward + backward.

    Parameters
    ----------
    model : Dict
        Model specification with ``input_dim``, ``width``, ``depth``.

    Returns
    -------
    float
        Approximate FLOPs per sample.
    """
    width = model.get("width", 512)
    depth = model.get("depth", 3)
    input_dim = model.get("input_dim", 784)
    # forward: input_dim*width + (depth-1)*width^2
    forward_flops = float(input_dim * width + (depth - 1) * width * width)
    # backward ≈ 2× forward
    return forward_flops * 3.0


def _eigenvalue_spectrum(n_modes: int, depth: int) -> NDArray:
    """Generate approximate NTK eigenvalue spectrum.

    Eigenvalues decay as ``1 / k^alpha`` where *alpha* grows with depth,
    reflecting increased spectral concentration in deeper networks.

    Parameters
    ----------
    n_modes : int
        Number of eigenvalue modes.
    depth : int
        Network depth (influences spectral decay rate).

    Returns
    -------
    NDArray
        Array of eigenvalues in descending order, shape ``(n_modes,)``.
    """
    alpha = 1.0 + 0.5 * math.log1p(depth)
    ks = np.arange(1, n_modes + 1, dtype=np.float64)
    return 1.0 / np.power(ks, alpha)


def _efficiency_factor(model: Dict) -> float:
    """Hardware efficiency factor (fraction of peak TFLOPS achieved).

    Larger models attain higher utilisation due to better arithmetic
    intensity.  Returns a value in [0.3, 0.5].

    Parameters
    ----------
    model : Dict
        Model specification.

    Returns
    -------
    float
        Efficiency factor in [0.3, 0.5].
    """
    width = model.get("width", 512)
    if width >= 2048:
        return 0.50
    if width >= 1024:
        return 0.45
    if width >= 512:
        return 0.40
    if width >= 256:
        return 0.35
    return 0.30


def _model_memory_gb(model: Dict) -> float:
    """Rough parameter-memory footprint in GB (float32)."""
    width = model.get("width", 512)
    depth = model.get("depth", 3)
    input_dim = model.get("input_dim", 784)
    n_params = input_dim * width + (depth - 1) * width * width + width
    # 4 bytes per param, ×3 for params + grads + optimizer state
    return n_params * 4.0 * 3.0 / (1024.0 ** 3)


# ======================================================================
# Core simulation
# ======================================================================

def simulate_training(
    model: Dict,
    dataset_stats: Dict,
    config: Dict,
) -> SimulatedRun:
    """Simulate a full training run using NTK spectral predictions.

    Parameters
    ----------
    model : Dict
        Model specification with keys ``input_dim``, ``width``, ``depth``,
        ``init_scale``.
    dataset_stats : Dict
        Dataset statistics with keys ``n_samples``, ``input_dim``,
        ``n_classes``, ``noise_level``.
    config : Dict
        Training configuration with keys ``lr``, ``epochs``, ``batch_size``,
        ``optimizer``.

    Returns
    -------
    SimulatedRun
        Simulated training trajectory and metadata.
    """
    width = model.get("width", 512)
    depth = model.get("depth", 3)
    init_scale = model.get("init_scale", 1.0)
    n_samples = dataset_stats.get("n_samples", 1000)
    n_classes = dataset_stats.get("n_classes", 10)
    noise_level = dataset_stats.get("noise_level", 0.1)

    lr = config.get("lr", 0.01)
    epochs = config.get("epochs", 100)
    batch_size = config.get("batch_size", 32)

    # Effective parameterisation ratio
    gamma = lr * (init_scale ** 2) / width

    # Determine regime
    if gamma < 0.8:
        regime = Regime.LAZY.value
    elif gamma > 1.2:
        regime = Regime.RICH.value
    else:
        regime = Regime.CRITICAL.value

    # NTK eigenvalue spectrum (use min of n_samples, width modes)
    n_modes = min(n_samples, width, 200)
    eigenvalues = _eigenvalue_spectrum(n_modes, depth)
    eigenvalues *= init_scale ** 2  # scale by initialisation

    # Steps per epoch
    steps_per_epoch = max(1, n_samples // batch_size)

    # Initial loss: cross-entropy at random ~ log(n_classes)
    initial_loss = math.log(n_classes) + 0.5 * noise_level

    # Build loss curve epoch by epoch
    loss_curve = np.empty(epochs, dtype=np.float64)
    accuracy_curve = np.empty(epochs, dtype=np.float64)

    convergence_epoch = -1
    convergence_threshold = 0.01 * initial_loss

    for ep in range(epochs):
        t = float((ep + 1) * steps_per_epoch)

        if regime == Regime.LAZY.value:
            # Lazy regime: each mode decays independently
            modal_residuals = np.exp(-eigenvalues * lr * t)
            loss = initial_loss * float(np.mean(modal_residuals))
        elif regime == Regime.RICH.value:
            # Rich regime: faster decay with oscillations
            freq = 0.1 * math.sqrt(gamma)
            modal_residuals = np.exp(-eigenvalues * lr * t * gamma) * (
                1.0 + 0.1 * np.sin(freq * t * eigenvalues)
            )
            modal_residuals = np.clip(modal_residuals, 0.0, None)
            loss = initial_loss * float(np.mean(modal_residuals))
        else:
            # Critical: interpolation
            lazy_res = np.exp(-eigenvalues * lr * t)
            rich_res = np.exp(-eigenvalues * lr * t * gamma) * (
                1.0 + 0.05 * np.sin(0.05 * t * eigenvalues)
            )
            rich_res = np.clip(rich_res, 0.0, None)
            alpha_crit = (gamma - 0.8) / 0.4
            loss = initial_loss * float(
                np.mean((1.0 - alpha_crit) * lazy_res + alpha_crit * rich_res)
            )

        loss = max(loss, noise_level * 0.01)
        loss_curve[ep] = loss

        # Accuracy: sigmoid mapping from loss
        normalised = loss / initial_loss
        base_acc = 1.0 / n_classes
        accuracy_curve[ep] = base_acc + (1.0 - base_acc) * (1.0 - normalised)
        accuracy_curve[ep] = min(accuracy_curve[ep], 1.0 - noise_level * 0.5)

        if convergence_epoch == -1 and loss < convergence_threshold:
            convergence_epoch = ep

    # Wall-time estimate (reference: single V100)
    flops_per_sample = _compute_flops_per_sample(model)
    total_flops = flops_per_sample * n_samples * epochs
    eff = _efficiency_factor(model)
    tflops = _GPU_TFLOPS["V100"] * 1e12
    wall_time = total_flops / (tflops * eff)

    return SimulatedRun(
        epochs=epochs,
        final_loss=float(loss_curve[-1]),
        final_accuracy=float(accuracy_curve[-1]),
        loss_curve=loss_curve,
        accuracy_curve=accuracy_curve,
        regime=regime,
        converged=convergence_epoch >= 0,
        convergence_epoch=convergence_epoch,
        wall_time_estimate=wall_time,
    )


# ======================================================================
# Grid search
# ======================================================================

def simulate_grid_search(
    model: Dict,
    param_grid: Dict,
    dataset_stats: Dict,
) -> GridResult:
    """Exhaustive grid search over hyperparameter combinations.

    Parameters
    ----------
    model : Dict
        Model specification.
    param_grid : Dict
        Maps parameter names to lists of candidate values, e.g.
        ``{"lr": [0.001, 0.01], "batch_size": [32, 64]}``.
    dataset_stats : Dict
        Dataset statistics.

    Returns
    -------
    GridResult
        Aggregated search results with parameter importance scores.
    """
    param_names = list(param_grid.keys())
    param_values = [param_grid[k] for k in param_names]

    configs: List[Dict] = []
    results: List[SimulatedRun] = []

    for combo in itertools.product(*param_values):
        cfg: Dict[str, Any] = dict(zip(param_names, combo))
        # Ensure required config keys have defaults
        cfg.setdefault("lr", 0.01)
        cfg.setdefault("epochs", 100)
        cfg.setdefault("batch_size", 32)
        cfg.setdefault("optimizer", "sgd")

        run = simulate_training(model, dataset_stats, cfg)
        configs.append(cfg)
        results.append(run)

    # Find best
    losses = [r.final_loss for r in results]
    best_idx = int(np.argmin(losses))

    # Parameter importance via variance decomposition
    loss_arr = np.array(losses, dtype=np.float64)
    total_var = float(np.var(loss_arr)) if len(loss_arr) > 1 else 1.0
    importances: Dict[str, float] = {}

    for p_idx, pname in enumerate(param_names):
        unique_vals = list(set(c[pname] for c in configs))
        if len(unique_vals) < 2 or total_var == 0.0:
            importances[pname] = 0.0
            continue
        group_means = []
        for val in unique_vals:
            group_losses = [
                losses[i] for i, c in enumerate(configs) if c[pname] == val
            ]
            group_means.append(np.mean(group_losses))
        between_var = float(np.var(group_means))
        importances[pname] = between_var / total_var

    # Normalise importances to sum to 1
    imp_sum = sum(importances.values())
    if imp_sum > 0:
        importances = {k: v / imp_sum for k, v in importances.items()}

    return GridResult(
        configs=configs,
        results=results,
        best_config=configs[best_idx],
        best_result=results[best_idx],
        param_importances=importances,
    )


# ======================================================================
# Hyperband
# ======================================================================

def simulate_hyperband(
    model: Dict,
    param_space: Dict,
    dataset_stats: Dict,
    budget: float = 100.0,
) -> HyperbandResult:
    """Hyperband-style successive-halving hyperparameter search.

    Parameters
    ----------
    model : Dict
        Model specification.
    param_space : Dict
        Maps parameter names to ``(min, max)`` tuples.  Learning rate
        (``"lr"``) is sampled in log-space; all others are linear.
    dataset_stats : Dict
        Dataset statistics.
    budget : float
        Maximum total epoch-budget across all brackets.

    Returns
    -------
    HyperbandResult
        Best configuration, budget usage, and per-bracket details.
    """
    eta = 3  # halving rate
    rng = np.random.RandomState(42)

    max_epochs = int(budget)
    s_max = max(int(math.floor(math.log(max_epochs) / math.log(eta))), 1)

    best_config: Dict = {}
    best_result: Optional[SimulatedRun] = None
    total_configs_evaluated = 0
    total_budget_used = 0.0
    bracket_results: List[Dict] = []

    def _sample_config() -> Dict:
        cfg: Dict[str, Any] = {}
        for name, (lo, hi) in param_space.items():
            if name == "lr":
                log_lo, log_hi = math.log10(lo), math.log10(hi)
                cfg[name] = float(10 ** rng.uniform(log_lo, log_hi))
            else:
                cfg[name] = float(lo + rng.random() * (hi - lo))
        cfg.setdefault("lr", 0.01)
        cfg.setdefault("epochs", max_epochs)
        cfg.setdefault("batch_size", 32)
        cfg.setdefault("optimizer", "sgd")
        return cfg

    for s in range(s_max, -1, -1):
        n = max(int(math.ceil(budget / max_epochs * (eta ** s) / (s + 1))), 1)
        r = max(int(max_epochs / (eta ** s)), 1)

        # Initial random configurations
        configs = [_sample_config() for _ in range(n)]
        bracket_budget = 0.0

        for i in range(s + 1):
            n_i = max(int(n / (eta ** i)), 1)
            r_i = min(int(r * (eta ** i)), max_epochs)

            runs: List[Tuple[Dict, SimulatedRun]] = []
            for cfg in configs[:n_i]:
                cfg_run = dict(cfg)
                cfg_run["epochs"] = r_i
                run = simulate_training(model, dataset_stats, cfg_run)
                runs.append((cfg, run))
                total_configs_evaluated += 1
                bracket_budget += r_i

            # Sort by final loss, keep top 1/eta
            runs.sort(key=lambda x: x[1].final_loss)
            keep = max(int(len(runs) / eta), 1)
            configs = [c for c, _ in runs[:keep]]

            # Track global best
            if runs:
                cand_cfg, cand_run = runs[0]
                if best_result is None or cand_run.final_loss < best_result.final_loss:
                    best_config = dict(cand_cfg)
                    best_result = cand_run

        total_budget_used += bracket_budget
        bracket_results.append({
            "bracket": s,
            "n_configs": n,
            "min_epochs": r,
            "max_epochs": min(int(r * (eta ** s)), max_epochs),
            "budget_used": bracket_budget,
        })

    if best_result is None:
        best_result = SimulatedRun()

    return HyperbandResult(
        best_config=best_config,
        best_result=best_result,
        configs_evaluated=total_configs_evaluated,
        total_budget_used=total_budget_used,
        bracket_results=bracket_results,
    )


# ======================================================================
# Cost estimation
# ======================================================================

def cost_estimator(
    model: Dict,
    dataset: Dict,
    cloud_config: Dict,
) -> CostEstimate:
    """Estimate cloud training cost for a given configuration.

    Parameters
    ----------
    model : Dict
        Model specification.
    dataset : Dict
        Dataset statistics (needs ``n_samples``).
    cloud_config : Dict
        Cloud configuration with keys ``provider``, ``gpu_type``,
        ``spot`` (bool).

    Returns
    -------
    CostEstimate
        Itemised cost estimate.
    """
    provider = cloud_config.get("provider", "generic")
    gpu_type = cloud_config.get("gpu_type", "V100")
    use_spot = cloud_config.get("spot", False)

    price_per_hour = _GPU_PRICE_PER_HOUR.get(gpu_type, 3.06)
    tflops = _GPU_TFLOPS.get(gpu_type, 15.7)

    epochs = 100  # default full-training budget
    n_samples = dataset.get("n_samples", 1000)

    flops_per_sample = _compute_flops_per_sample(model)
    total_flops = flops_per_sample * n_samples * epochs
    eff = _efficiency_factor(model)
    compute_seconds = total_flops / (tflops * 1e12 * eff)
    hours = compute_seconds / 3600.0

    on_demand = hours * price_per_hour
    spot_cost = on_demand * _SPOT_DISCOUNT

    # Breakdown: compute dominates; add minor storage/network
    storage = 0.05 * hours  # ~$0.05/hr for attached storage
    network = 0.02 * hours  # minimal egress

    breakdown = {
        "compute": on_demand,
        "storage": storage,
        "network": network,
    }
    total_on_demand = on_demand + storage + network
    total_spot = spot_cost + storage + network

    return CostEstimate(
        provider=provider,
        gpu_type=gpu_type,
        hours=hours,
        cost_usd=total_on_demand,
        spot_cost_usd=total_spot,
        breakdown=breakdown,
    )


# ======================================================================
# Time estimation
# ======================================================================

def time_estimator(
    model: Dict,
    dataset: Dict,
    hardware: str,
) -> TimeEstimate:
    """Estimate training wall-clock time on specified hardware.

    Parameters
    ----------
    model : Dict
        Model specification.
    dataset : Dict
        Dataset statistics (needs ``n_samples``).
    hardware : str
        Hardware identifier (``"V100"``, ``"A100"``, ``"H100"``,
        ``"T4"``, ``"CPU"``).

    Returns
    -------
    TimeEstimate
        Time estimate with throughput and bottleneck analysis.
    """
    tflops = _GPU_TFLOPS.get(hardware, 0.1)
    gpu_mem = _GPU_MEMORY_GB.get(hardware, 16.0)

    n_samples = dataset.get("n_samples", 1000)
    epochs = 100

    flops_per_sample = _compute_flops_per_sample(model)
    total_flops = flops_per_sample * n_samples * epochs
    eff = _efficiency_factor(model)
    compute_seconds = total_flops / (tflops * 1e12 * eff)
    hours = compute_seconds / 3600.0

    throughput = n_samples / max(compute_seconds / epochs, 1e-9)

    # Bottleneck analysis
    mem_required = _model_memory_gb(model)
    if mem_required > gpu_mem * 0.8:
        bottleneck = "memory"
        # Memory-bound: reduce effective utilisation
        utilization = min(eff * 0.7, 0.35)
        hours /= max(utilization / eff, 0.5)
    else:
        bottleneck = "compute"
        utilization = eff

    return TimeEstimate(
        hardware=hardware,
        hours=hours,
        throughput_samples_per_sec=throughput,
        bottleneck=bottleneck,
        gpu_utilization=utilization,
    )
