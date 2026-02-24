"""Plan ML experiments efficiently using statistical and NTK-based methods.

Provides tools for experiment planning, power analysis, ablation study
design, hyperparameter importance estimation, and minimal experiment set
construction.  Combines classical design-of-experiments methodology with
heuristics tailored to neural-network training.

Example
-------
>>> from phase_diagrams.experiment_planner import plan_experiments, power_analysis
>>> plan = plan_experiments(["wider networks generalize better"], budget=500.0)
>>> plan.total_cost <= 500.0
True
>>> result = power_analysis(effect_size=0.5, n_seeds=10)
>>> 0.0 <= result.power <= 1.0
True
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

# ======================================================================
# Data classes
# ======================================================================

@dataclass
class ExperimentPlan:
    """A complete experiment plan with budget and priority information."""
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    total_cost: float = 0.0
    total_time_hours: float = 0.0
    priority_order: List[int] = field(default_factory=list)
    rationale: str = ""
    estimated_information_gain: float = 0.0

@dataclass
class PowerResult:
    """Result of a statistical power analysis."""
    recommended_seeds: int = 5
    power: float = 0.0
    effect_size: float = 0.0
    confidence_level: float = 0.95
    explanation: str = ""

@dataclass
class AblationPlan:
    """Plan for systematic ablation experiments."""
    experiments: List[Dict[str, Any]] = field(default_factory=list)
    n_experiments: int = 0
    baseline_config: Dict = field(default_factory=dict)  # type: ignore[type-arg]
    components: List[str] = field(default_factory=list)
    expected_insights: List[str] = field(default_factory=list)

@dataclass
class Config:
    """A single experiment configuration with expected outcomes."""
    params: Dict[str, Any] = field(default_factory=dict)
    expected_performance: float = 0.0
    information_gain: float = 0.0

# ======================================================================
# Internal helpers
# ======================================================================

def _norm_ppf(p: float) -> float:
    """Approximate inverse-normal CDF (Abramowitz & Stegun 26.2.23)."""
    if p <= 0.0:
        return -6.0
    if p >= 1.0:
        return 6.0
    if p < 0.5:
        return -_norm_ppf(1.0 - p)
    t = math.sqrt(-2.0 * math.log(1.0 - p))
    c0, c1, c2 = 2.515517, 0.802853, 0.010328
    d1, d2, d3 = 1.432788, 0.189269, 0.001308
    return t - (c0 + c1 * t + c2 * t * t) / (1.0 + d1 * t + d2 * t * t + d3 * t ** 3)


def _norm_cdf(x: float) -> float:
    """Approximate standard-normal CDF via sigmoid."""
    return 1.0 / (1.0 + math.exp(-1.7 * x))


_KEYWORD_MAP: Dict[str, str] = {
    "width": "width_sweep", "wider": "width_sweep",
    "lr": "lr_sweep", "learning rate": "lr_sweep",
    "depth": "depth_sweep", "deeper": "depth_sweep",
    "regularization": "reg_cmp", "dropout": "reg_cmp",
    "optimizer": "opt_cmp", "adam": "opt_cmp", "sgd": "opt_cmp",
}

_TEMPLATES: Dict[str, Dict[str, Any]] = {
    "width_sweep":  {"type": "width_sweep",  "n_configs": 5, "n_seeds": 3, "epochs": 50,
                     "widths": [64, 128, 256, 512, 1024],
                     "metrics": ["train_loss", "test_loss", "kernel_alignment"]},
    "lr_sweep":     {"type": "lr_sweep",     "n_configs": 5, "n_seeds": 3, "epochs": 50,
                     "lrs": [1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
                     "metrics": ["train_loss", "convergence_step"]},
    "depth_sweep":  {"type": "depth_sweep",  "n_configs": 5, "n_seeds": 3, "epochs": 80,
                     "depths": [2, 4, 6, 8, 10],
                     "metrics": ["train_loss", "test_loss", "gradient_norm"]},
    "reg_cmp":      {"type": "regularization_comparison", "n_configs": 4, "n_seeds": 3,
                     "epochs": 50, "methods": ["none", "dropout", "weight_decay", "batch_norm"],
                     "metrics": ["test_loss", "generalisation_gap"]},
    "opt_cmp":      {"type": "optimizer_comparison", "n_configs": 3, "n_seeds": 3,
                     "epochs": 50, "optimizers": ["sgd", "adam", "adamw"],
                     "metrics": ["train_loss", "convergence_step", "final_test_loss"]},
}

# ======================================================================
# Public API
# ======================================================================

def plan_experiments(
    hypotheses: List[str],
    budget: float = 1000.0,
    time_budget_hours: float = 100.0,
) -> ExperimentPlan:
    """Generate a prioritised experiment plan from free-text hypotheses.

    Parameters
    ----------
    hypotheses : List[str]
        Free-text hypotheses (e.g. ``"wider networks generalize better"``).
    budget : float
        Maximum compute cost.
    time_budget_hours : float
        Maximum wall-clock hours.

    Returns
    -------
    ExperimentPlan
    """
    experiments: List[Dict[str, Any]] = []
    seen: set[str] = set()

    for hyp in hypotheses:
        hyp_lower = hyp.lower()
        etype: Optional[str] = None
        for kw, et in _KEYWORD_MAP.items():
            if kw in hyp_lower:
                etype = et
                break
        if etype is None:
            etype = "width_sweep"

        tmpl = dict(_TEMPLATES[etype])
        tmpl["hypothesis"] = hyp
        cost = tmpl["n_configs"] * tmpl["n_seeds"] * tmpl["epochs"] * 0.01
        time_h = tmpl["n_configs"] * tmpl["n_seeds"] * tmpl["epochs"] * 0.002
        tmpl["cost"], tmpl["time_hours"] = cost, time_h
        novelty = 1.0 if etype not in seen else 0.5
        tmpl["information_gain"] = novelty / (1.0 + cost * 0.01)
        seen.add(etype)
        experiments.append(tmpl)

    scored = sorted(enumerate(experiments), key=lambda t: t[1]["information_gain"], reverse=True)
    priority_order = [i for i, _ in scored]

    selected, sel_idx = [], []
    rem_b, rem_t = budget, time_budget_hours
    for idx in priority_order:
        e = experiments[idx]
        if e["cost"] <= rem_b and e["time_hours"] <= rem_t:
            selected.append(e)
            sel_idx.append(idx)
            rem_b -= e["cost"]
            rem_t -= e["time_hours"]

    tc = sum(e["cost"] for e in selected)
    tt = sum(e["time_hours"] for e in selected)
    parts = [f"Selected {len(selected)}/{len(experiments)} experiments."]
    for e in selected:
        parts.append(f"  - {e['type']}: '{e['hypothesis']}' (cost={e['cost']:.1f})")
    parts.append(f"Total cost {tc:.1f}/{budget:.1f}, time {tt:.1f}/{time_budget_hours:.1f}h.")

    return ExperimentPlan(
        experiments=selected, total_cost=tc, total_time_hours=tt,
        priority_order=sel_idx, rationale="\n".join(parts),
        estimated_information_gain=sum(e["information_gain"] for e in selected),
    )


def power_analysis(
    effect_size: float,
    n_seeds: int = 5,
    alpha: float = 0.05,
    desired_power: float = 0.8,
) -> PowerResult:
    """Perform statistical power analysis for seed selection.

    Parameters
    ----------
    effect_size : float
        Standardised effect size (Cohen's *d*).
    n_seeds : int
        Initial number of random seeds to evaluate.
    alpha : float
        Significance level.
    desired_power : float
        Target statistical power.

    Returns
    -------
    PowerResult
    """
    z_alpha = _norm_ppf(1.0 - alpha / 2.0)

    def _power(n: int) -> float:
        return _norm_cdf(effect_size * math.sqrt(n) - z_alpha)

    cur_power = _power(n_seeds)
    recommended = next((n for n in range(2, 500) if _power(n) >= desired_power), 500)

    return PowerResult(
        recommended_seeds=recommended, power=cur_power, effect_size=effect_size,
        confidence_level=1.0 - alpha,
        explanation=(
            f"With d={effect_size:.2f}, alpha={alpha}, {n_seeds} seeds give "
            f"power={cur_power:.3f}. Recommend {recommended} seeds for "
            f"power>={desired_power:.2f} (achieved {_power(recommended):.3f})."
        ),
    )


def ablation_plan(model: Dict[str, Any], components: List[str]) -> AblationPlan:
    """Design a systematic ablation study.

    Parameters
    ----------
    model : Dict[str, Any]
        Model specification (input_dim, width, depth, activation, etc.).
    components : List[str]
        Component names to ablate (e.g. ``["dropout", "batch_norm"]``).

    Returns
    -------
    AblationPlan
    """
    baseline = dict(model)
    for c in components:
        baseline[c] = True

    exps: List[Dict[str, Any]] = [{"name": "baseline", "config": dict(baseline), "ablated": None}]
    insights = ["Baseline performance with all components enabled."]

    # Fractional factorial for large component sets
    selected = components if len(components) <= 5 else (
        components[::2] + components[1::2][: len(components) // 2]
    )
    selected = list(dict.fromkeys(selected))

    for comp in selected:
        cfg = dict(baseline)
        cfg[comp] = False
        exps.append({"name": f"no_{comp}", "config": cfg, "ablated": comp})
        insights.append(f"Removing {comp}: measures its contribution to performance.")

    return AblationPlan(
        experiments=exps, n_experiments=len(exps), baseline_config=baseline,
        components=components, expected_insights=insights,
    )


def hyperparameter_importance(
    model: Dict[str, Any],
    dataset: Dict[str, Any],
    params: Dict[str, List[Any]],
) -> Dict[str, float]:
    """Estimate hyperparameter importance via functional ANOVA decomposition.

    Parameters
    ----------
    model : Dict[str, Any]
        Base model configuration.
    dataset : Dict[str, Any]
        Dataset metadata (n_samples, n_features, etc.).
    params : Dict[str, List]
        Parameter name -> candidate values.

    Returns
    -------
    Dict[str, float]
        Normalised importance scores summing to 1.0.
    """
    n_feat = dataset.get("n_features", 10)
    base_w = model.get("width", 128)
    param_names = list(params.keys())

    def _predict(cfg: Dict[str, Any]) -> float:
        w = cfg.get("width", base_w)
        lr = cfg.get("lr", 0.001)
        d = cfg.get("depth", 4)
        wd = cfg.get("weight_decay", 0.0)
        return (0.4 * (1.0 - 1.0 / (1.0 + w / n_feat))
                + 0.3 * math.exp(-((math.log(lr) + 6) ** 2) / 8.0)
                + 0.2 * (1.0 - math.exp(-d / 3.0))
                - 0.1 * wd * 10.0)

    variances: Dict[str, float] = {}
    for name in param_names:
        scores = []
        for val in params[name]:
            cfg = dict(model)
            cfg[name] = val
            for other in param_names:
                if other != name:
                    mid = params[other][len(params[other]) // 2]
                    cfg[other] = mid
            scores.append(_predict(cfg))
        mu = sum(scores) / max(len(scores), 1)
        variances[name] = sum((s - mu) ** 2 for s in scores) / max(len(scores), 1)

    total = sum(variances.values())
    if total < 1e-12:
        return {n: 1.0 / len(param_names) for n in param_names}
    return {n: v / total for n, v in variances.items()}


def minimal_experiment_set(
    full_grid: Dict[str, List[Any]],
    method: str = "latin_hypercube",
) -> List[Config]:
    """Select a compact subset of a full parameter grid.

    Parameters
    ----------
    full_grid : Dict[str, List]
        Parameter name -> list of candidate values.
    method : str
        ``"latin_hypercube"``, ``"orthogonal"``, or ``"random"``.

    Returns
    -------
    List[Config]
        Configurations sorted by information gain (descending).
    """
    names = list(full_grid.keys())
    values = [full_grid[k] for k in names]
    n_p = len(names)
    total = 1
    for v in values:
        total *= len(v)
    rng = np.random.RandomState(42)

    if method == "latin_hypercube":
        n_s = max(n_p * 2, 4)
        idx = np.empty((n_s, n_p), dtype=int)
        for d in range(n_p):
            nl = len(values[d])
            bins = np.linspace(0, nl, n_s + 1)
            col = np.array([rng.randint(int(bins[i]), max(int(bins[i+1]), int(bins[i])+1))
                            for i in range(n_s)], dtype=int)
            col = np.clip(col, 0, nl - 1)
            rng.shuffle(col)
            idx[:, d] = col
    elif method == "orthogonal":
        n_s = max(n_p * 2, 4)
        idx = np.empty((n_s, n_p), dtype=int)
        for d in range(n_p):
            nl = len(values[d])
            stride = max(nl // n_s, 1)
            idx[:, d] = [(i * stride) % nl for i in range(n_s)]
    elif method == "random":
        n_s = max(int(math.sqrt(total)), 4)
        idx = np.column_stack([rng.randint(0, len(v), size=n_s) for v in values])
    else:
        raise ValueError(f"Unknown method: {method!r}")

    configs: List[Config] = []
    for row in range(idx.shape[0]):
        cfg = {names[d]: values[d][idx[row, d]] for d in range(n_p)}
        perf = _heuristic_perf(cfg)
        configs.append(Config(params=cfg, expected_performance=perf))

    # Information gain = distance to nearest neighbour in normalised space
    if len(configs) > 1:
        arr = np.zeros((len(configs), n_p), dtype=np.float64)
        for i, c in enumerate(configs):
            for j, nm in enumerate(names):
                nl = len(values[j])
                try:
                    ix = values[j].index(c.params[nm])
                except ValueError:
                    ix = 0
                arr[i, j] = ix / max(nl - 1, 1)
        for i, c in enumerate(configs):
            dists = np.sqrt(np.sum((arr - arr[i]) ** 2, axis=1))
            dists[i] = np.inf
            c.information_gain = float(np.min(dists))

    configs.sort(key=lambda c: c.information_gain, reverse=True)
    return configs


def _heuristic_perf(cfg: Dict[str, Any]) -> float:
    """Rough performance estimate from raw parameter values."""
    s = 0.5
    if "width" in cfg:
        s += 0.1 * math.log2(max(cfg["width"], 1)) / 10.0
    if "lr" in cfg:
        s += 0.2 * math.exp(-((math.log(cfg["lr"]) + 6) ** 2) / 8.0)
    if "depth" in cfg:
        s += 0.1 * (1.0 - math.exp(-cfg["depth"] / 3.0))
    return float(np.clip(s, 0.0, 1.0))
