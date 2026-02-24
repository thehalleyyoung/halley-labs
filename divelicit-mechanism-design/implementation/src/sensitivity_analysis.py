"""Sensitivity analysis for DivFlow hyperparameters.

Analyzes the sensitivity of DivFlow mechanism properties to:
1. Sinkhorn regularization epsilon
2. Quality weight lambda
3. Z3 grid resolution
4. Selection size k
5. Number of candidates n

Reports how violation rates, coverage, and welfare change across
parameter sweeps with confidence intervals.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from .transport import sinkhorn_divergence, sinkhorn_candidate_scores
from .coverage import clopper_pearson_ci, bootstrap_ci


@dataclass
class SensitivityResult:
    """Result of a single parameter sweep point."""
    param_name: str
    param_value: float
    violation_rate: float
    violation_ci: Tuple[float, float]
    topic_coverage: float
    mean_quality: float
    welfare: float
    n_trials: int


def _greedy_select(embs, quals, k, quality_weight, reg):
    """Greedy welfare-maximizing selection."""
    n = len(quals)
    ref = embs.copy()
    selected = []

    for _ in range(min(k, n)):
        if len(selected) == 0:
            best_j = int(np.argmax(quals))
        else:
            scores = sinkhorn_candidate_scores(
                embs, embs[selected], ref, reg=reg, n_iter=50,
            )
            combined = np.full(n, -np.inf)
            for j in range(n):
                if j not in selected:
                    s_max = max(abs(scores).max(), 1e-10)
                    d_norm = scores[j] / s_max
                    combined[j] = (1 - quality_weight) * d_norm + quality_weight * quals[j]
            best_j = int(np.argmax(combined))
        selected.append(best_j)
    return selected


def _welfare(embs, quals, selected, quality_weight, reg):
    """Compute welfare W(S)."""
    if not selected:
        return 0.0
    sel_embs = embs[selected]
    sdiv = sinkhorn_divergence(sel_embs, embs, reg=reg, n_iter=50)
    q_sum = sum(quals[i] for i in selected)
    return -(1.0 - quality_weight) * sdiv + quality_weight * q_sum


def _compute_vcg_payments(embs, quals, selected, k, quality_weight, reg):
    """Compute VCG payments for selected agents."""
    n = len(quals)
    payments = []
    for agent in selected:
        others = [j for j in selected if j != agent]
        w_others = _welfare(embs, quals, others, quality_weight, reg)
        candidates = [j for j in range(n) if j != agent]
        best_without = []
        for _ in range(min(k, len(candidates))):
            best_j, best_gain = -1, -np.inf
            for j in candidates:
                if j in best_without:
                    continue
                trial = best_without + [j]
                gain = _welfare(embs, quals, trial, quality_weight, reg) - \
                       _welfare(embs, quals, best_without, quality_weight, reg)
                if gain > best_gain:
                    best_gain = gain
                    best_j = j
            if best_j >= 0:
                best_without.append(best_j)
        w_without = _welfare(embs, quals, best_without, quality_weight, reg)
        payments.append(float(max(w_without - w_others, 0.0)))
    return payments


def _test_ic_violation_rate(embs, quals, k, quality_weight, reg,
                            n_trials=200, seed=42):
    """Test IC violation rate for given parameters."""
    rng = np.random.RandomState(seed)
    n = len(quals)
    selected = _greedy_select(embs, quals, k, quality_weight, reg)
    payments = _compute_vcg_payments(embs, quals, selected, k, quality_weight, reg)

    violations = 0
    max_gain = 0.0
    for _ in range(n_trials):
        agent = rng.randint(n)
        true_q = quals[agent]
        if agent in selected:
            pos = selected.index(agent)
            truthful_u = true_q - payments[pos]
        else:
            truthful_u = 0.0

        fake_q = rng.uniform(0, 1)
        dev_quals = quals.copy()
        dev_quals[agent] = fake_q
        dev_selected = _greedy_select(embs, dev_quals, k, quality_weight, reg)
        dev_payments = _compute_vcg_payments(embs, dev_quals, dev_selected, k, quality_weight, reg)

        if agent in dev_selected:
            pos = dev_selected.index(agent)
            dev_u = true_q - dev_payments[pos]
        else:
            dev_u = 0.0

        gain = dev_u - truthful_u
        if gain > 1e-8:
            violations += 1
            max_gain = max(max_gain, gain)

    ci = clopper_pearson_ci(violations, n_trials)
    return violations / n_trials, ci, max_gain


def sensitivity_sinkhorn_epsilon(embs, quals, topics, k=5,
                                  quality_weight=0.3,
                                  epsilons=None,
                                  n_trials=200, seed=42):
    """Sweep Sinkhorn regularization epsilon."""
    if epsilons is None:
        epsilons = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0]

    results = []
    for eps in epsilons:
        selected = _greedy_select(embs, quals, k, quality_weight, eps)
        viol_rate, viol_ci, _ = _test_ic_violation_rate(
            embs, quals, k, quality_weight, eps, n_trials, seed
        )
        sel_topics = set(topics[i] for i in selected)
        n_topics = len(set(topics))
        w = _welfare(embs, quals, selected, quality_weight, eps)
        results.append(SensitivityResult(
            param_name="sinkhorn_epsilon",
            param_value=eps,
            violation_rate=viol_rate,
            violation_ci=viol_ci,
            topic_coverage=len(sel_topics) / n_topics,
            mean_quality=float(np.mean(quals[selected])),
            welfare=float(w),
            n_trials=n_trials,
        ))
    return results


def sensitivity_quality_weight(embs, quals, topics, k=5,
                                reg=0.1,
                                lambdas=None,
                                n_trials=200, seed=42):
    """Sweep quality weight lambda."""
    if lambdas is None:
        lambdas = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]

    results = []
    for lam in lambdas:
        selected = _greedy_select(embs, quals, k, lam, reg)
        viol_rate, viol_ci, _ = _test_ic_violation_rate(
            embs, quals, k, lam, reg, n_trials, seed
        )
        sel_topics = set(topics[i] for i in selected)
        n_topics = len(set(topics))
        w = _welfare(embs, quals, selected, lam, reg)
        results.append(SensitivityResult(
            param_name="quality_weight",
            param_value=lam,
            violation_rate=viol_rate,
            violation_ci=viol_ci,
            topic_coverage=len(sel_topics) / n_topics,
            mean_quality=float(np.mean(quals[selected])),
            welfare=float(w),
            n_trials=n_trials,
        ))
    return results


def sensitivity_selection_size(embs, quals, topics,
                                quality_weight=0.3, reg=0.1,
                                k_values=None,
                                n_trials=200, seed=42):
    """Sweep selection size k."""
    n = len(quals)
    if k_values is None:
        k_values = [3, 5, 8, 10, 15]

    results = []
    for k in k_values:
        if k >= n:
            continue
        selected = _greedy_select(embs, quals, k, quality_weight, reg)
        viol_rate, viol_ci, _ = _test_ic_violation_rate(
            embs, quals, k, quality_weight, reg, n_trials, seed
        )
        sel_topics = set(topics[i] for i in selected)
        n_topics = len(set(topics))
        w = _welfare(embs, quals, selected, quality_weight, reg)
        results.append(SensitivityResult(
            param_name="selection_size_k",
            param_value=float(k),
            violation_rate=viol_rate,
            violation_ci=viol_ci,
            topic_coverage=len(sel_topics) / n_topics,
            mean_quality=float(np.mean(quals[selected])),
            welfare=float(w),
            n_trials=n_trials,
        ))
    return results


def sensitivity_z3_grid_resolution(embs, quals, k=3,
                                    quality_weight=0.3, reg=0.1,
                                    resolutions=None, seed=42):
    """Sweep Z3 grid resolution and measure certification rate and soundness gap."""
    if resolutions is None:
        resolutions = [5, 10, 15, 20, 25]

    results = []
    for grid_res in resolutions:
        n = len(quals)
        grid = np.linspace(0.0, 1.0, grid_res)
        grid_spacing = 1.0 / (grid_res - 1)

        # Lipschitz estimation per agent
        lipschitz_max = 0.0
        selected_base = _greedy_select(embs, quals, k, quality_weight, reg)

        for agent in range(n):
            gains = []
            for q_val in grid:
                test_quals = quals.copy()
                test_quals[agent] = q_val
                sel_test = _greedy_select(embs, test_quals, k, quality_weight, reg)
                pay_test = _compute_vcg_payments(embs, test_quals, sel_test, k, quality_weight, reg)
                if agent in sel_test:
                    pos = sel_test.index(agent)
                    u = quals[agent] - pay_test[pos]
                else:
                    u = 0.0
                gains.append(u)

            diffs = [abs(gains[i+1] - gains[i]) / grid_spacing
                     for i in range(len(gains) - 1)]
            if diffs:
                lipschitz_max = max(lipschitz_max, max(diffs))

        # IC violation test at this grid resolution
        viol_rate, viol_ci, max_gain = _test_ic_violation_rate(
            embs, quals, k, quality_weight, reg, n_trials=100, seed=seed,
        )
        soundness_gap = lipschitz_max * grid_spacing

        results.append({
            "grid_resolution": grid_res,
            "grid_spacing": grid_spacing,
            "violation_rate": viol_rate,
            "violation_ci": list(viol_ci),
            "lipschitz_max": lipschitz_max,
            "soundness_gap": soundness_gap,
            "soundness_holds": soundness_gap < 1e-4,
        })
    return results


def bootstrap_sensitivity(embs, quals, topics, k=5,
                          quality_weight=0.3, reg=0.1,
                          n_bootstrap=500, seed=42):
    """Bootstrap CIs for key sensitivity metrics across resampled data."""
    rng = np.random.RandomState(seed)
    n = len(quals)
    coverages = []
    viol_rates = []
    welfares = []

    for b in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        b_embs = embs[idx]
        b_quals = quals[idx]
        b_topics = topics[idx]

        selected = _greedy_select(b_embs, b_quals, k, quality_weight, reg)
        sel_topics = set(b_topics[i] for i in selected)
        n_total_topics = len(set(b_topics))
        coverages.append(len(sel_topics) / max(n_total_topics, 1))
        welfares.append(_welfare(b_embs, b_quals, selected, quality_weight, reg))

        # Quick IC check (5 trials per bootstrap for speed)
        violations = 0
        payments = _compute_vcg_payments(b_embs, b_quals, selected, k, quality_weight, reg)
        for _ in range(5):
            agent = rng.randint(n)
            true_q = b_quals[agent]
            if agent in selected:
                pos = selected.index(agent)
                truthful_u = true_q - payments[pos]
            else:
                truthful_u = 0.0
            fake_q = rng.uniform(0, 1)
            dev_quals = b_quals.copy()
            dev_quals[agent] = fake_q
            dev_sel = _greedy_select(b_embs, dev_quals, k, quality_weight, reg)
            dev_pay = _compute_vcg_payments(b_embs, dev_quals, dev_sel, k, quality_weight, reg)
            if agent in dev_sel:
                pos = dev_sel.index(agent)
                dev_u = true_q - dev_pay[pos]
            else:
                dev_u = 0.0
            if dev_u - truthful_u > 1e-8:
                violations += 1
        viol_rates.append(violations / 5.0)

    coverages = np.array(coverages)
    viol_rates = np.array(viol_rates)
    welfares = np.array(welfares)

    return {
        "coverage_mean": float(np.mean(coverages)),
        "coverage_ci_95": [float(np.percentile(coverages, 2.5)),
                           float(np.percentile(coverages, 97.5))],
        "violation_rate_mean": float(np.mean(viol_rates)),
        "violation_rate_ci_95": [float(np.percentile(viol_rates, 2.5)),
                                  float(np.percentile(viol_rates, 97.5))],
        "welfare_mean": float(np.mean(welfares)),
        "welfare_ci_95": [float(np.percentile(welfares, 2.5)),
                          float(np.percentile(welfares, 97.5))],
        "n_bootstrap": n_bootstrap,
    }


def full_sensitivity_analysis(embs, quals, topics, k=5,
                               quality_weight=0.3, reg=0.1,
                               n_trials=100, seed=42):
    """Run complete sensitivity analysis across all hyperparameters.

    Returns dict with results for each parameter sweep, including
    Z3 grid resolution sweep and bootstrap CIs.
    """
    results = {}

    # 1. Sinkhorn epsilon
    eps_results = sensitivity_sinkhorn_epsilon(
        embs, quals, topics, k=k, quality_weight=quality_weight,
        epsilons=[0.01, 0.05, 0.1, 0.2, 0.5, 1.0],
        n_trials=n_trials, seed=seed,
    )
    results["sinkhorn_epsilon"] = [
        {
            "value": r.param_value,
            "violation_rate": r.violation_rate,
            "violation_ci": list(r.violation_ci),
            "topic_coverage": r.topic_coverage,
            "mean_quality": r.mean_quality,
            "welfare": r.welfare,
        }
        for r in eps_results
    ]

    # 2. Quality weight lambda
    lam_results = sensitivity_quality_weight(
        embs, quals, topics, k=k, reg=reg,
        lambdas=[0.0, 0.1, 0.3, 0.5, 0.7, 1.0],
        n_trials=n_trials, seed=seed,
    )
    results["quality_weight"] = [
        {
            "value": r.param_value,
            "violation_rate": r.violation_rate,
            "violation_ci": list(r.violation_ci),
            "topic_coverage": r.topic_coverage,
            "mean_quality": r.mean_quality,
            "welfare": r.welfare,
        }
        for r in lam_results
    ]

    # 3. Selection size k
    k_results = sensitivity_selection_size(
        embs, quals, topics, quality_weight=quality_weight, reg=reg,
        k_values=[3, 5, 8, 10],
        n_trials=n_trials, seed=seed,
    )
    results["selection_size"] = [
        {
            "value": r.param_value,
            "violation_rate": r.violation_rate,
            "violation_ci": list(r.violation_ci),
            "topic_coverage": r.topic_coverage,
            "mean_quality": r.mean_quality,
            "welfare": r.welfare,
        }
        for r in k_results
    ]

    # 4. Z3 grid resolution sweep
    results["z3_grid_resolution"] = sensitivity_z3_grid_resolution(
        embs, quals, k=min(k, len(quals) - 1),
        quality_weight=quality_weight, reg=reg,
        resolutions=[5, 10, 15, 20],
        seed=seed,
    )

    # 5. Bootstrap CIs across resampled data
    results["bootstrap_stability"] = bootstrap_sensitivity(
        embs, quals, topics, k=k,
        quality_weight=quality_weight, reg=reg,
        n_bootstrap=200, seed=seed,
    )

    return results
